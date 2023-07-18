from typing import Tuple

import torch
from transformers import BertModel, BertTokenizer
import sys
sys.path.append('..')
from Datasets.dataIterSandR import Batch


class BERTRanker(torch.nn.Module):
    """A simple BERT ranker that uses the CLS output as a score. The inputs are constructed using an approximated
    k-hot sample, i.e. only k sentences are considered. The correspondign inputs are multiplied with their weights.

    Args:
        bert_type (str): Pre-trained BERT model type
        bert_dim (int): BERT output dimension
        dropout (float): Dropout value
        freeze (bool, optional): Don't update ranker weights during training. Defaults to False.
    """
    def __init__(self, bert_type: str, bert_dim: int, dropout: float, bert_cache: str, freeze: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_type, return_dict=True, cache_dir=bert_cache)
        self.dropout = torch.nn.Dropout(dropout)
        self.classification = torch.nn.Linear(bert_dim, 1)

        tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=bert_cache)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.max_len = 512

        for p in self.parameters():
            p.requires_grad = not freeze


    def _get_single_input(self, query_in: torch.LongTensor, doc_in: torch.LongTensor, lengths: torch.IntTensor,
                          weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct a single BERT input sequence.

        Args:
            query_in (torch.LongTensor): Query input IDs
            doc_in (torch.LongTensor): Document input IDs
            lengths (Sequence[int]): Passage lengths
            weights (torch.Tensor): Passage weights

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: BERT inputs and weights
        """
        # device for new tensors
        dev = query_in.device
        cls_tensor = torch.as_tensor([self.cls_id], device=dev)
        sep_tensor = torch.as_tensor([self.sep_id], device=dev)

        # keep track of sequence length to construct padding, mask and token type IDs
        in_ids = [cls_tensor, query_in, sep_tensor]
        running_seq_len = len(query_in) + 2

        # token types are 0 up until (including) the 1st SEP
        tt_ids = [torch.as_tensor([0] * running_seq_len, device=dev)]

        # CLS/query part -- set all weights to 1, as we always keep the query
        in_weights = [torch.as_tensor([1.0] * running_seq_len, device=dev)]

        # document part -- drop passages with weight of 0, copy the weights for all
        idx = 0
        for length, weight in zip(lengths, weights):
            # we should never see padding here as the number of weights is equal to the number of passages
            assert length > 0

            next_idx = idx + length
            if weight == 1.0:
                in_ids.append(doc_in[idx:next_idx])

                # token types are 1 up until (including) the 2nd SEP
                tt_ids.append(torch.as_tensor([1] * length, device=dev))

                in_weights.extend([weight.unsqueeze(0)] * length)
                running_seq_len += length
            idx = next_idx

        # last token should be SEP
        in_ids.append(sep_tensor)
        running_seq_len += 1

        # mask is 1 up until the 2nd SEP
        mask = [torch.as_tensor([1.0] * running_seq_len, device=dev)]

        # if the sequence is not max length, pad
        remaining = self.max_len - min(running_seq_len, self.max_len)
        if remaining > 0:
            in_ids.append(torch.as_tensor([self.pad_id] * remaining, device=dev))

            # token types and mask are 0 after the 2st SEP
            mask.append(torch.as_tensor([0.0] * remaining, device=dev))

        # these need one more for the separator
        tt_ids.append(torch.as_tensor([0] * (remaining + 1), device=dev))
        in_weights.append(torch.as_tensor([1.0] * (remaining + 1), device=dev))

        # truncate to maximum length
        in_ids = torch.cat(in_ids)[:self.max_len]
        mask = torch.cat(mask)[:self.max_len]
        tt_ids = torch.cat(tt_ids)[:self.max_len]
        in_weights = torch.cat(in_weights)[:self.max_len]

        # make sure lengths match
        assert in_ids.shape[-1] == mask.shape[-1] == tt_ids.shape[-1] == in_weights.shape[-1] == self.max_len
        return in_ids, mask, tt_ids, in_weights

    def forward(self, batch: Batch, weights: torch.Tensor) -> torch.FloatTensor:
        """Classify a batch of inputs, using the k highest scored passages as BERT input.

        Args:
            batch (Batch): The input batch
            weights (torch.Tensor): The weights

        Returns:
            torch.FloatTensor: Relevance scores for each input
        """
        batch_in_ids, batch_masks, batch_tt_ids, batch_weights = [], [], [], []
        for query_in, query_length, doc_in, doc_length, lengths, weights_ in zip(*batch, weights):

            # remove padding
            query_in = query_in[:query_length]
            doc_in = doc_in[:doc_length]

            # create BERT inputs
            in_ids, mask, tt_ids, weights_in = self._get_single_input(query_in, doc_in, lengths, weights_)
            batch_in_ids.append(in_ids)
            batch_masks.append(mask)
            batch_tt_ids.append(tt_ids)
            batch_weights.append(weights_in)

        # create a batch of BERT inputs
        batch_in_ids = torch.stack(batch_in_ids)
        batch_masks = torch.stack(batch_masks)
        batch_tt_ids = torch.stack(batch_tt_ids)
        batch_weights = torch.stack(batch_weights)

        # create actual input by multiplying weights with input embeddings
        batch_emb = self.bert.embeddings(input_ids=batch_in_ids)
        bert_in_batch = batch_emb * batch_weights.unsqueeze(-1).expand_as(batch_emb)

        bert_out = self.bert(inputs_embeds=bert_in_batch, token_type_ids=batch_tt_ids,
                             attention_mask=batch_masks)
        cls_out = bert_out['last_hidden_state'][:, 0]
        return self.classification(self.dropout(cls_out))
