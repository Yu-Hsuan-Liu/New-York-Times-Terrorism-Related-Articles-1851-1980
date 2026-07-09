# -*- coding: utf-8 -*-
"""
Rerun LDA coherence analysis for K=2..15 on balanced subsample (597/period).
Saves scores to nlp_results/LDA_Coherence_Scores.csv
"""
import random, os, csv
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

random.seed(123)
np.random.seed(123)

OUT_DIR = 'C:/Users/tosea/claude_test_project/nlp_results'
os.makedirs(OUT_DIR, exist_ok=True)

print('Loading parquet...')
df = pd.read_parquet('D:/Jupyter/NYT_API/combined_processed_df.parquet')
df = df[df['clean_lemmatized_phrased'].notna()].copy()

label_col = 'label'
text_col  = 'clean_lemmatized_phrased'

periods = sorted(df[label_col].unique())
counts  = {p: len(df[df[label_col]==p]) for p in periods}
min_n   = min(counts.values())
print(f'Min n: {min_n}')

print(f'Sampling {min_n} per period...')
sampled = pd.concat([
    df[df[label_col] == p].sample(min_n, random_state=123, replace=False)
    for p in periods
], ignore_index=True)
print(f'Total: {len(sampled)}')

texts = [str(t).split() for t in sampled[text_col]]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)
print(f'Dictionary size: {len(dictionary)}')
corpus = [dictionary.doc2bow(t) for t in texts]

results = []
for k in range(2, 16):
    print(f'Fitting K={k}...', flush=True)
    lda = LdaModel(
        corpus, num_topics=k, id2word=dictionary,
        passes=20, random_state=123
    )
    cm = CoherenceModel(
        model=lda, texts=texts, dictionary=dictionary, coherence='c_v', processes=1
    )
    score = cm.get_coherence()
    results.append((k, round(score, 4)))
    print(f'  K={k}: c_v = {score:.4f}', flush=True)

out_csv = os.path.join(OUT_DIR, 'LDA_Coherence_Scores.csv')
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['K', 'c_v_coherence'])
    writer.writerows(results)

best_k, best_score = max(results, key=lambda x: x[1])
print(f'\nBest K: {best_k} (c_v = {best_score:.4f})')
print(f'Saved: {out_csv}')
print('\nFull results:')
for k, s in results:
    marker = ' <-- best' if k == best_k else ''
    print(f'  K={k:2d}: {s:.4f}{marker}')
