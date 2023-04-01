# Pyserini utilities,data info, tf, df, bm25, etc.
import math
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from typing import List, Tuple, Dict
from pathlib import Path
from nltk.corpus import stopwords
STOP = stopwords.words('english')

def load_searcher(directory: Path) -> SimpleSearcher:
    searcher = SimpleSearcher(directory)
    return searcher


def loader_index(directory: Path) -> IndexReader:
    index_reader = IndexReader(directory)
    return index_reader

def load_analyzer(stemmer: str = 'krovetz'):
    analyzer = Analyzer(get_lucene_analyzer(stemmer=stemmer))
    return analyzer

def term_freqs(indexer: IndexReader, term: str, analyzer=None) -> Tuple[float, float]:
    df, cf = indexer.get_term_counts(term, analyzer=analyzer)   # without analyzing/stemming, search the current type of term directly.
    return df, cf


def term_posts(indexer: IndexReader, term: str):   # -> [post.docid, post.tf, post.positions] 
    # term freqs and positions
    positions = indexer.get_postings_list(term)
    return positions


def doc_tfidf(indexer: IndexReader, docid: str) -> Tuple[float, float, float]:  
    # tf: {'terms':  tf}
    N = indexer.stats()['documents']  # the number of documents
    tf = indexer.get_document_vector(docid)
    df = {term: (term_freqs(indexer, term))[0] for term in tf.keys()}
    tf_idf = {term: tf[term] * math.log(N/(df[term] + 1)) for term in tf.keys() } 
    return tf, df, tf_idf


def doc_bm25(indexer: IndexReader, docid: str) -> Dict[str, float]:
    tf = indexer.get_document_vector(docid)
    bm25_vector = {term: indexer.compute_bm25_term_weight(docid, term, analyzer=None) for term in tf.keys()}
    return bm25_vector


def doc_query_score(indexer: IndexReader, query: str, docids: List[str]) -> List[float]:
    scores = [indexer.compute_query_document_score(docid, query) for docid in docids]
    return scores


def _index_search(searcher: SimpleSearcher, query: str) -> List[Tuple[str, float]]:
    hits = searcher.search(query, 100)   # hard decide as 100 for now.
    hit_docid_score = [(h.docid, h.score) for h in hits]
    return hit_docid_score


def get_candidates(index_reader: IndexReader, docid: str, topk: int) -> Dict:
    tf, df, tfidf = doc_tfidf(index_reader, docid)
    sorted_x = sorted(tfidf.items(), key=lambda kv: kv[1], reverse=True)
    sorted_nostop = [(a, b) for a, b in sorted_x if a not in STOP]
    return dict(sorted_nostop[:topk])