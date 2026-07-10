# Analyzing Terrorism in Historical News: A Natural Language Processing Approach

Replication materials for the manuscript.

The repository has four parts. `raw_data_processing/` holds the notebooks that collected and assembled the 1851-2019 NYT corpus (metadata scraping, ProQuest PDF OCR for the pre-1981 articles, merging); its own README documents the collection pipeline step by step. `code/` holds the analysis scripts, `results/` their outputs, and `figures/` the manuscript figures. `data/` carries the article lists.

Several source files exceed GitHub's 100 MB limit and are not included: the full-text files `nyt_online_texts.csv` (427 MB), `NYT_text_1851_1980.csv` (108 MB), and `nyt_terrorism_pdftexts_proquest2.csv` (100 MB), and the two processed parquet corpora (457 MB each). Contact the corresponding author for copies, or rebuild them with the collection notebooks (post-1980 full texts require ProQuest access; see Section 3.1 of the manuscript).

To rerun the analysis: `nyt_main_training.py` trains the per-period Word2Vec models, `run_per_period_lda.py` and `run_lda_coherence.py` fit and score the topic models, `rerun_lda_rf_option_a.py` runs the LDA plus Random Forest pipeline, and the remaining scripts cover robustness checks (unified min_count, Procrustes alignment, bootstrap rank stability, concordance validation) and figure generation. `REPRODUCE.md` gives the full run order and maps each script to the tables and figures it produces.

Requires Python 3.8+ with gensim, scikit-learn, pandas, numpy, matplotlib, seaborn, and pyarrow. A CUDA GPU helps with the Word2Vec training.

Please cite the manuscript if you use these materials.
