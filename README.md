# Analyzing Terrorism in Historical News: A Natural Language Processing Approach

Replication materials for the manuscript.

## Repository Structure

```
repository root
├── raw_data_processing/   # Data collection & preprocessing notebooks
│   ├── README.md                          # Pipeline documentation
│   ├── NYT_WEB_SCRAPER.ipynb             # Step 1: Scrape article metadata from NYT
│   ├── scrape_information_of_news_under_category_of_terrorism_on_NYT_website.ipynb
│   │                                      # Step 2: Consolidate list + scrape online texts
│   ├── combine_all_terrorism_news_and_scrape_texts_1850_1980(ProQuestNYU).ipynb
│   │                                      # Step 3: ProQuest PDF OCR (pre-1981)
│   ├── 1850_1980_combine_test1.ipynb      # Step 4: Merge pre-1981 data
│   └── combine_all_terrorism_news_and_scrape_texts_1850_2019.ipynb
│                                          # Step 5: Assemble full 1851-2019 corpus
├── data/                  # Source and intermediate data
│   ├── total_nyt_terrorism_news1850-2019.csv       # Master article list (177K rows)
│   ├── total_nyt_terrorism_news1850-1980_raw.csv   # Pre-1980 article list
│   ├── nyt_terrorism_pdftexts_proquest2.csv        # ProQuest OCR texts (22K rows) [not in repo; see Large Data Files]
│   ├── NYT_text_1851_1980.csv                      # Merged pre-1981 texts [not in repo; see Large Data Files]
│   └── nyt_online_texts.csv                        # Online scraped texts (92K rows) [not in repo; see Large Data Files]
├── code/                  # All analysis scripts
│   ├── my_functions_NYT_gpu.py          # Shared utility functions
│   ├── nyt_main_training.py             # W2V model training (CBOW & SkipGram)
│   ├── run_per_period_lda.py            # Per-period LDA topic modeling
│   ├── run_lda_coherence.py             # LDA coherence scoring
│   ├── run_normalized_lda.py            # Normalized LDA topic proportions
│   ├── rerun_lda_rf_option_a.py         # LDA + RF pipeline (Option A)
│   ├── rf_no_leakage.py                 # Leak-free Random Forest classifier
│   ├── run_rf_bootstrap_main.py         # RF bootstrap confidence intervals
│   ├── run_rf_comparison.py             # RF vectorizer comparison (CV vs TF-IDF)
│   ├── run_temporal_alignment.py        # Procrustes temporal alignment
│   ├── w2v_klan_sensitivity.py          # KKK normalization sensitivity check
│   ├── w2v_robustness_check.py          # W2V robustness (unified min_count)
│   ├── make_visualization.py            # LDA / RF / Procrustes visualizations
│   ├── make_w2v_visualization.py        # W2V heatmap visualizations
│   ├── generate_rf_figures.py           # RF figure generation
│   ├── regenerate_rf_figures.py         # RF figure regeneration
│   ├── task_a_bootstrap.py              # Bootstrap rank stability (Supplementary Table S9)
│   ├── task_a2_category_bootstrap.py    # Category-level bootstrap, 1851-1900
│   └── task_b_e_cooccurrence.py         # Concordance validation (Supplementary Table S10)
├── results/               # Model outputs and analysis results
│   ├── FINAL_ALL_MODELS_RESULTS.csv
│   ├── Dynamic_LDA_Results_Updated.csv
│   ├── LDA_Coherence_Scores.csv
│   ├── LDA_Topic_Keywords.csv
│   ├── LDA_Topic_Proportions.csv
│   ├── LDA_Normalized_TopicProportions.csv
│   ├── LDA_Normalized_Keywords.txt
│   ├── LDA_Topic_Chart.png
│   ├── RF_Final_PerClass.csv
│   ├── RF_Feature_Importance.csv
│   ├── RF_Confusion_Matrix.csv
│   ├── RF_Comparison.csv
│   ├── RF_PerClass_Metrics.csv
│   ├── RF_Final_Bootstrap_CI.txt
│   ├── RF_MainModel_Bootstrap_CI.txt
│   ├── RF_Classification_Report.txt
│   ├── RF_Metrics.txt
│   ├── RF_CountVectorizer_FullReport.txt
│   ├── RF_TF_IDF_FullReport.txt
│   ├── RF_PerClass_Report.txt
│   ├── W2V_Robustness_Comparison.csv
│   ├── W2V_Robustness_FullResults.csv
│   ├── W2V_Robustness_Summary.txt
│   ├── Procrustes_Displacement_Results.csv
│   ├── Procrustes_Alignment_Summary.txt
│   ├── Context_1851-1900_terroris_klan.txt
│   ├── Context_1901-1930_terroris_night_rider.txt
│   ├── bootstrap_ranks.csv                          # Replicate-level bootstrap ranks (B=50)
│   ├── bootstrap_ranks_summary.csv                  # Token-level stability (Suppl. Table S9)
│   ├── bootstrap_summary.txt
│   ├── category_bootstrap_1851_1900.csv             # Category-level stability, 1851-1900
│   ├── category_bootstrap_summary.txt
│   ├── predicate_cooccurrence.csv                   # Concordance counts (Suppl. Table S10)
│   ├── predicate_cooccurrence_snippets.txt          # Dated concordance lines
│   ├── kwic_bulgar_1901_1930.txt                    # KWIC lines for 'bulgar', 1901-1930
│   ├── kwic_bulgar_near_terroris.txt
│   ├── lda_coherence_log.txt
│   ├── normalized_lda_log.txt
│   ├── rf_comparison_log.txt
│   └── sample_no_leak/                  # Trained RF model artifacts
│       ├── vectorizer.pkl
│       └── my_model.pkl
└── figures/               # Manuscript figures
    ├── Figure1_W2V_Heatmap_CBOW.png
    ├── Figure2_LDA_Heatmap.png
    ├── Figure2_LDA_Topics.png
    ├── Figure2_Procrustes_Displacement.png
    ├── Figure3_W2V_Heatmap.png
    ├── Figure4_RF_Confusion_Heatmap.png
    ├── Figure4_RF_ConfusionMatrix.png
    ├── Figure5_RF_F1_ByPeriod.png
    └── Figure6_RF_FeatureImportance.png
```

## Large Data Files

The following files exceed GitHub's 100 MB file size limit and require Git LFS or can be obtained by contacting the corresponding author:

| File | Size | Description |
|------|------|-------------|
| `data/nyt_online_texts.csv` | 427 MB | Online scraped article texts (post-1980) |
| `data/NYT_text_1851_1980.csv` | 108 MB | Merged pre-1981 article texts |
| `data/nyt_terrorism_pdftexts_proquest2.csv` | 100 MB | ProQuest OCR texts |
| `combined_processed_df.parquet` | 457 MB | Processed NLP-ready corpus (not included) |
| `combined_processed_df_updated.parquet` | 457 MB | Corpus with LDA topic labels (not included) |

## Analysis Pipeline

The scripts should be run in the following order:

1. **W2V Training**: `nyt_main_training.py` — Trains Word2Vec CBOW and SkipGram models per historical period using `combined_processed_df.parquet`
2. **LDA Topic Modeling**: `run_per_period_lda.py` — Per-period LDA with optimized K
3. **LDA Coherence**: `run_lda_coherence.py` — Computes coherence scores for LDA models
4. **LDA + RF Pipeline**: `rerun_lda_rf_option_a.py` — Produces `combined_processed_df_updated.parquet` with topic labels, then runs Random Forest
5. **Normalized LDA**: `run_normalized_lda.py` — Normalized topic proportions
6. **RF Leak-Free**: `rf_no_leakage.py` — Random Forest with no data leakage
7. **RF Bootstrap**: `run_rf_bootstrap_main.py` — Bootstrap confidence intervals
8. **RF Comparison**: `run_rf_comparison.py` — CountVectorizer vs TF-IDF comparison
9. **Temporal Alignment**: `run_temporal_alignment.py` — Procrustes alignment for semantic displacement
10. **Robustness Checks**: `w2v_robustness_check.py`, `w2v_klan_sensitivity.py`
11. **Bootstrap & Concordance Validation**: `task_a_bootstrap.py`, `task_a2_category_bootstrap.py`, `task_b_e_cooccurrence.py` — Rank-stability bootstrap (Supplementary Table S9) and article-level concordance validation (Supplementary Table S10)
12. **Visualizations**: `make_visualization.py`, `make_w2v_visualization.py`, `generate_rf_figures.py`

## Requirements

- Python 3.8+
- gensim, scikit-learn, pandas, numpy, matplotlib, seaborn, pyarrow
- CUDA-capable GPU recommended for W2V training

## License

Please cite the manuscript if you use these materials.
