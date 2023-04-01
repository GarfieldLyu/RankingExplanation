import csv
import sys
sys.path.append('models')
sys.path.append('Datasets')
sys.path.append('utilities')

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from pathlib import Path
import math
import random
import numpy as np
from itertools import combinations
from typing import Dict, Tuple, List, Any

import explainers
from explainers import get_explainer
from utilities import utility, kendalltau_concord


csv.field_size_limit(sys.maxsize)
project_dir = Path.cwd()
seed = 100

analyzer = utility.load_analyzer()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Explain(object):
    def __init__(self, hparams: Dict[str, Any]): 

        """ Init the base explainer object, load the model to be explained and the data object.
            Args: hparams: a dictionary of hyperparameters and necessary objects.
                index_dir: the directory of the corpus index, pre-built using pyserini.
                RankModel: the model to be explained.
                InferenceDataset: the dataset object for inference, without y label.
                dataIterate: the dataIterate object for inference data, for easier doc perturbation and reranking.
                queries: a dictionary of queries, with q_id as key and query as value.
        """
        print('Initing indexes...')  
        self.index_reader = utility.loader_index(hparams['index_dir'])
        self.model = hparams['RankModel']
        self.InferenceDataset = hparams['InferenceDataset']
        self.dataIterate = hparams['dataIterate']
        self.queries = hparams['queries']
        
    def _init_query(self, q_id: str, rank_scores: bool= False):
        self.InferenceDataset.__init_q_docs__(q_id, self.queries[q_id])
        self.InferenceDataset.query_tokens = [q for q in analyzer.analyze(self.InferenceDataset.query) if q not in utility.STOP]
        print(f'query: {self.queries[q_id]}')
        if rank_scores:
            # get the prediction scores of each doc in the list
            prediction = self._rank_docs(self.InferenceDataset.query, self.InferenceDataset.top_docs)
            self.InferenceDataset.prediction = prediction
            self.InferenceDataset.rank = np.argsort(-self.InferenceDataset.prediction) # sort doc index from high to low

    def _rank_docs(self, query:str, docs: List[str], batch_size = 64):
        inputs_data = self.dataIterate.CustomDataset(query, docs, self.InferenceDataset.tokenizer, device)
        inputs_iter = DataLoader(inputs_data, batch_size = batch_size, collate_fn=getattr(inputs_data, 'collate_fn', None))
        prediction = np.array([])
        with torch.no_grad():
            for i, batch in enumerate(inputs_iter):
                out = self.model(batch).detach().cpu().squeeze(-1).numpy()
                prediction = np.append(prediction, out)
        return prediction

    def refine_candidates_by_perturb(self, replaced_tokens: Dict[str, float], doc_id: str, doc:str) -> Dict[str, float]:
        """ Compute the candidate tokens which influence the document the most by masking the token in the document and comparing prediction diff."""
        replaced_tokens = list(replaced_tokens.keys())
        input_orig = self.InferenceDataset.__buildFromDoc__(doc)
        score_orig = self.model(input_orig).detach().cpu().item()
        new_docs = []
        for replace_token in replaced_tokens:
            new_docs.append(doc.replace(f' {replace_token} ', '[UNK]'))   # keep blank before and after the token to avoid characters.
        
        prediction = self._rank_docs(self.InferenceDataset.query, new_docs)
        score_diff = abs(score_orig - prediction)
    
        refined = dict((k, v) for k, v in zip(replaced_tokens, score_diff))
        refined = sorted(refined.items(), key=lambda kv: kv[1], reverse=True)
        return dict(refined)

    def refine_candidates_by_bm25(self, replaced_tokens: Dict[str, float], doc_id: str, doc: str) -> Dict[str, float]:
        replaced_tokens = list(replaced_tokens.keys())
        bm25_scores = [self.index_reader.compute_bm25_term_weight(doc_id, q, analyzer=None) for q in replaced_tokens]
        term_score = dict((k, v) for k, v in zip(replaced_tokens, bm25_scores))
        term_score = sorted(term_score.items(), key=lambda kv: kv[1], reverse=True)
        return dict(term_score)

    def get_candidates_reranker(self, q_id: str, topd: int, topk: int, topr: int, method: str='bm25') -> Dict[str, float]:
        if method == 'bm25':
            refine_method = self.refine_candidates_by_bm25
        elif method == 'perturb':
            refine_method = self.refine_candidates_by_perturb
        elif method == 'None':
            refine_method = lambda x, y, z: x   # only return the first argument.
        else:
            raise ValueError('Invalid candidates selecting method.')
        candidates_scores = {}
        self._init_query(q_id)
        for doc_id, doc in tqdm(zip(self.InferenceDataset.top_docs_id[:topd], self.InferenceDataset.top_docs[:topd]), desc='Perturb each doc...'):
            candidates_tfidf = utility.get_candidates(self.index_reader, doc_id, topk)
            candidates_refined = refine_method(candidates_tfidf, doc_id, doc)
            for k, v in candidates_refined.items():
                if k in candidates_scores:
                    candidates_scores[k] += [v]
                else:
                    candidates_scores[k] = [v]
        # average scores
        for k, v in candidates_scores.items():
            candidates_scores[k] = sum(v)/len(v)
        candidates_scores = sorted(candidates_scores.items(), key=lambda kv: kv[1], reverse=True)
        refined_candidates = dict(candidates_scores[:topr])
        return refined_candidates

    def sample_doc_pair(self, ranked: int=20, m: int=500, style: str='random', tolerance: float=2.0) -> List[Tuple[int, int]]:
        if style == 'random':
            pairs = list(combinations(range(self.InferenceDataset.length), 2))
        elif style == 'topk_random':
            assert(ranked <= self.InferenceDataset.length)
            ranked_list = list(range(ranked))
            tail_list = list(range(ranked, self.InferenceDataset.length))
            pairs = [(a, b) for a in ranked_list for b in tail_list]
        else:
            raise ValueError(f'Not supported style {style}')
        # filter our pairs with prediction scores diffence < tolerance, e.g., 0.01
        rank = np.argsort(-self.InferenceDataset.prediction)   
        probs_diff = np.array([self.InferenceDataset.prediction[rank[h]] - self.InferenceDataset.prediction[rank[l]] for h, l in pairs])
        valid_index = list(np.where(probs_diff >= tolerance)[0])
        #valid_index = list(np.where(probs_diff >= 0.01)[0])
        pairs = [pairs[i] for i in valid_index]
        random.seed(seed)
        if len(pairs) < m:
            m = len(pairs)
        pairs = random.sample(pairs, m)  
        return pairs

    def build_matrix(self, candidates: List[str], pairs: List[Tuple[int]], EXP_model: str='language_model') -> List[List[float]]:
        # need to find candidates cooccur in both docs.
        explainer = get_explainer(EXP_model)
        matrix = []
        print(f'Sampled {len(pairs)} doc pairs')

        for rank_h_id, rank_l_id in tqdm(pairs, desc="building matrix for doc pairs..."):
            weight = 1 + math.log(rank_l_id - rank_h_id)    
            doc_h_id = self.InferenceDataset.rank[rank_h_id]
            doc_l_id = self.InferenceDataset.rank[rank_l_id]
            doc_h = self.InferenceDataset.top_docs[doc_h_id]
            doc_l = self.InferenceDataset.top_docs[doc_l_id]
            s_h = explainer(candidates, doc_h, analyzer)
            s_l = explainer(candidates, doc_l, analyzer )
            concordance = (np.array(s_h) - np.array(s_l)) * np.array(weight)
            matrix.append(concordance.tolist())
        # reshape matrix to candidate dimension first. 
        matrix = np.array(matrix).transpose(1, 0).tolist()
        return matrix

    def evaluate_fidelity(self, expansions: List[str], EXP_model: List[str], top_k: int=10, vote: int=2, tolerance: float=2.0) -> Tuple[float, float, float, float]:
        print('Kendalltau evaluation...')
        prediction_orig = self.InferenceDataset.prediction.copy()     
        rank = self.InferenceDataset.rank.copy()
        pred_orig_topk = prediction_orig[rank[:top_k]]
        if isinstance(EXP_model, str):
            # single-ranker
            EXP_model = [EXP_model]   
        prediction_new = explainers.multi_rank(EXP_model, expansions, self.InferenceDataset.top_docs, analyzer)
        
        if len(EXP_model) <= 1:
            pred_new_topk = prediction_new[rank[:top_k]]
            correl_g = kendalltau_concord.kendalltau(prediction_orig, prediction_new).correlation
            correl_l = kendalltau_concord.kendalltau(pred_orig_topk, pred_new_topk).correlation
            correl_tg = kendalltau_concord.kendalltau_gap(prediction_orig, prediction_new, tolerance)
            correl_tl = kendalltau_concord.kendalltau_gap(pred_orig_topk, pred_new_topk, tolerance)
        
        else:
            # multi vote, consider all explainers.
            pred_new_topk_all = [pred[rank[:top_k]] for pred in prediction_new]
            correl_g = kendalltau_concord.coverage_multi(prediction_orig, prediction_new, vote=vote, tolerance=0)
            correl_l = kendalltau_concord.coverage_multi(pred_orig_topk, pred_new_topk_all, vote=vote, tolerance=0)
            correl_tg = kendalltau_concord.coverage_multi(prediction_orig, prediction_new, vote=vote, tolerance=tolerance)
            correl_tl = kendalltau_concord.coverage_multi(pred_orig_topk, pred_new_topk_all, vote=vote, tolerance=tolerance)
        return correl_g, correl_l, correl_tg, correl_tl 

    
    
