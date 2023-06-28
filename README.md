# RankingExplanation
This is the source code for paper "Listwise Explanations for Ranking Models using Multiple Explainers".
To cite the paper:
```
@inproceedings{lyu2023listwise,
  title={Listwise Explanations for Ranking Models Using Multiple Explainers},
  author={Lyu, Lijun and Anand, Avishek},
  booktitle={European Conference on Information Retrieval},
  pages={653--668},
  year={2023},
  organization={Springer}
}
```

This entry of running explanation methods is `explain_run.py`, before starting, one needs to make sure the necessary packages, pre-computed index, model to be explained and the dataset are ready. For each dataset, we pre-compute the index with pyserini package, please refer to the [official repository](https://github.com/castorini/pyserini) for more details. We use the index to implement basic statistic functions (e.g., tf-idf). For our method multiplex, we use [geno solver](https://www.geno-project.org) to approximate the ILP problem. No installation is required, a python code will be generated from the website, based on the mathematic equations provided.  

The explanation methods (baselines and multiplex) included in this repository are generally pipeline-based with three major steps:
- generate explanation term candidates from the list of documents, for each individual query.   
- generate the coverage matrix using one explainer (i.e. a simple ranking model) for the candidates from the previous step. If multiple explainers are used, the same number of matrices are also needed.

The two steps are defined in `explain_base.py`, for the last step of generating explanation terms and evaluate the fidelity, one can choose different methods by passing the argument `optimize_method` in `explain_run.py`. It will call optimize methods defined under `utilities`. Please note that for all methods optimized by geno solver, the python code has to be pre-generated from the official site, the mathematic equations used are recorded in the start of the code (e.g. the objectives of our method defined in the paper can be found in geno_solver_multi.py).