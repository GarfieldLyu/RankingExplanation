import torch
from torchtext.vocab import Vocab
from argparse import ArgumentParser
from typing import Any, Dict, Union

import sys
sys.path.append("/home/lyu/ExpRank/")
sys.path.append("/home/lyu/ExpRank/Datasets/")
import glove
from base_model import BaseRanker
from dataIterGlove import PointwiseTrainDataset, PairwiseTrainDataset, ValTestDataset, Batch, PointwiseTrainBatch, PairwiseTrainBatch


class QALSTMRanker(BaseRanker):
    """QA-LSTM for passage ranking using GloVe embeddings.
    Args:
        hparams (Dict[str, Any]): All model hyperparameters
        vocab (Vocab): Vocabulary
        rr_k (int, optional): Compute MRR@k. Defaults to 10.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 16.
        training_mode (str, optional): Training mode, 'pointwise' or 'pairwise'. Defaults to 'pairwise'.
    """
    def __init__(self, hparams: Dict[str, Any], rr_k: int = 10, num_workers: int = 16, training_mode: str = 'pairwise'):

        #'Init vocabulary first...'
        self.vocab = glove.GloveTokenizer(hparams['data_file'], hparams['vocab_dir'])
        if training_mode == 'pointwise':
            train_ds = PointwiseTrainDataset(hparams['data_file'], hparams['train_file_pointwise'], self.vocab)
        else:
            assert training_mode == 'pairwise'
            train_ds = PairwiseTrainDataset(hparams['data_file'], hparams['train_file_pairwise'], self.vocab)
        val_ds = ValTestDataset(hparams['data_file'], hparams['val_file'], self.vocab)
        test_ds = ValTestDataset(hparams['data_file'], hparams['test_file'], self.vocab)
        uses_ddp = 'ddp' in hparams['distributed_backend']
        super().__init__(hparams, train_ds, val_ds, test_ds, hparams['loss_margin'], hparams['batch_size'], rr_k, num_workers, uses_ddp)

        pad_id = self.vocab.pad_id
        emb_dim = self.vocab.vocab.vectors[0].shape[0]
        self.embedding = torch.nn.Embedding.from_pretrained(self.vocab.vocab.vectors, freeze=False, padding_idx=pad_id)
        self.lstm = torch.nn.LSTM(emb_dim, hparams['hidden_dim'], batch_first=True, bidirectional=True)

        # attention weights
        self.W_am = torch.nn.Linear(hparams['hidden_dim'] * 2, hparams['hidden_dim'] * 2)
        self.W_qm = torch.nn.Linear(hparams['hidden_dim'] * 2, hparams['hidden_dim'] * 2)
        self.w_ms = torch.nn.Linear(hparams['hidden_dim'] * 2, 1)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

        self.dropout = torch.nn.Dropout(hparams['dropout'])
        self.cos_sim = torch.nn.CosineSimilarity()

    def _encode(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Embed and encode a batch of padded sequences using the shared LSTM.
        Args:
            inputs (torch.Tensor): The padded input sequences
            lengths (torch.Tensor): The sequence lengths
        Returns:
            torch.Tensor: The LSTM outputs
        """
        input_embed = self.embedding(inputs)
        input_seqs = torch.nn.utils.rnn.pack_padded_sequence(input_embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(input_seqs)
        out_seqs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return out_seqs

    def _max_pool(self, lstm_outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Perform max-pooling on the LSTM outputs, masking padding tokens.
        Args:
            lstm_outputs (torch.Tensor): LSTM output sequences
            lengths (torch.Tensor): Sequence lengths
        Returns:
            torch.Tensor: Maximum along dimension 1
        """
        num_sequences, max_seq_len, num_hidden = lstm_outputs.shape

        # create mask
        rng = torch.arange(max_seq_len, device=lstm_outputs.device).unsqueeze(0).expand(num_sequences, -1)
        rng = rng.unsqueeze(-1).expand(-1, -1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, max_seq_len, -1)
        mask = rng < lengths

        # set padding outputs to -inf so they dont affect max pooling
        lstm_outputs_clone = lstm_outputs.clone()
        lstm_outputs_clone[~mask] = float('-inf')
        return torch.max(lstm_outputs_clone, dim=1)[0]

    def _attention(self, query_outputs_pooled: torch.Tensor, doc_outputs: torch.Tensor, doc_lengths: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted outputs for a batch of queries and documents, masking padding tokens.
        Args:
            query_outputs_pooled (torch.Tensor): Encoded queries after pooling
            doc_outputs (torch.Tensor): Encoded documents
            doc_lengths (torch.Tensor): Document lengths
        Returns:
            torch.Tensor: Attention-weighted outputs
        """
        # doc_outputs has shape (num_docs, max_seq_len, 2 * hidden_dim)
        # query_outputs_pooled has shape (num_docs, 2 * hidden_dim)
        # expand its shape so they match
        max_seq_len = doc_outputs.shape[1]
        m = self.tanh(self.W_am(doc_outputs) + self.W_qm(query_outputs_pooled).unsqueeze(1).expand(-1, max_seq_len, -1))
        wm = self.w_ms(m)

        # mask the padding tokens before computing the softmax by setting the corresponding values to -inf
        mask = torch.arange(max_seq_len, device=wm.device)[None, :] < doc_lengths[:, None]
        wm[~mask] = float('-inf')

        s = self.softmax(wm)
        return doc_outputs * s

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return the similarities for all query and document pairs.
        Args:
            batch (Batch): The input batch
        Returns:
            torch.Tensor: The similarities
        """
        self.lstm.flatten_parameters()
        queries, query_lengths, docs, doc_lengths = batch

        query_outputs = self._encode(queries, query_lengths)
        query_outputs_pooled = self._max_pool(query_outputs, query_lengths)

        doc_outputs = self._encode(docs, doc_lengths)
        attention = self._attention(query_outputs_pooled, doc_outputs, doc_lengths)
        attention_pooled = self._max_pool(attention, doc_lengths)

        return self.cos_sim(self.dropout(query_outputs_pooled), self.dropout(attention_pooled)).unsqueeze(1)

    def configure_optimizers(self) -> torch.optim.Adam:
        """Create an Adam optimizer.
        Returns:
            torch.optim.Adam: The optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def training_step(self, batch: Union[PointwiseTrainBatch, PairwiseTrainBatch], batch_idx: int) -> torch.Tensor:
        """Train a single batch. In pairwise training mode, we use the similarity directly without sigmoid.
        Args:
            batch (Union[PointwiseTrainBatch, PairwiseTrainBatch]): A training batch, depending on the mode
            batch_idx (int): Batch index
        Returns:
            torch.Tensor: Training loss
        """
        if self.training_mode == 'pointwise':
            inputs, labels = batch
            loss = self.bce(self(inputs).flatten(), labels.flatten())
        else:
            pos_inputs, neg_inputs = batch
            pos_outputs = self(pos_inputs)
            neg_outputs = self(neg_inputs)
            loss = torch.mean(torch.clamp(self.loss_margin - pos_outputs + neg_outputs, min=0))
        self.log('train_loss', loss)
        return loss

    @staticmethod
    def add_model_specific_args(ap: ArgumentParser):
        """Add model-specific arguments to the parser.
        Args:
            ap (ArgumentParser): The parser
        """
        ap.add_argument('--hidden_dim', type=int, default=256, help='The hidden dimensions throughout the model')
        ap.add_argument('--dropout', type=float, default=0.5, help='Dropout percentage')
        ap.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        ap.add_argument('--loss_margin', type=float, default=0.2, help='Margin for pairwise loss')
        ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
