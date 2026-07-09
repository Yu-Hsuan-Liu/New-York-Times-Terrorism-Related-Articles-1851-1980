# -*- coding: utf-8 -*-
"""
Bootstrap 95% CI for the main saved model (sample/my_model.pkl + vectorizer.pkl).
Uses same balanced subsample and split as the per-class metrics run (seed=123).
"""
import pickle, random, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

random.seed(123)
np.random.seed(123)

OUT_DIR = 'C:/Users/tosea/claude_test_project/nlp_results'

print('Loading parquet...')
df = pd.read_parquet('D:/Jupyter/NYT_API/combined_processed_df.parquet')
df = df[df['clean_lemmatized_phrased'].notna()].copy()

label_col = 'label'
text_col  = 'clean_lemmatized_phrased'

periods = sorted(df[label_col].unique())
min_n   = min(len(df[df[label_col]==p]) for p in periods)
print(f'Min n: {min_n}')

sampled = pd.concat([
    df[df[label_col] == p].sample(min_n, random_state=123, replace=False)
    for p in periods
], ignore_index=True)

X = sampled[text_col].astype(str)
y = sampled[label_col]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

print('Loading saved vectorizer and model...')
with open('D:/Jupyter/NYT_API/sample/vectorizer.pkl', 'rb') as f:
    vec = pickle.load(f)
with open('D:/Jupyter/NYT_API/sample/my_model.pkl', 'rb') as f:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = pickle.load(f)

X_test_v = vec.transform(X_test)
y_pred   = model.predict(X_test_v)

f1_main = f1_score(y_test, y_pred, average='weighted')
print(f'Main model Weighted F1 on this test split: {f1_main:.4f}')

# Bootstrap CI
print('Running bootstrap CI (1000 iterations)...')
rng = np.random.default_rng(123)
n = len(y_test)
y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_pred)
boot_f1s = []
for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    boot_f1s.append(f1_score(
        y_test_arr[idx], y_pred_arr[idx],
        average='weighted', zero_division=0
    ))

ci_lo = np.percentile(boot_f1s, 2.5)
ci_hi = np.percentile(boot_f1s, 97.5)
print(f'95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]')

out = os.path.join(OUT_DIR, 'RF_MainModel_Bootstrap_CI.txt')
with open(out, 'w') as f:
    f.write(f'Main model (sample/my_model.pkl + sample/vectorizer.pkl)\n')
    f.write(f'Weighted F1: {f1_main:.4f}\n')
    f.write(f'95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n')
    f.write(f'Based on {n} test samples, 1000 bootstrap iterations\n')
print(f'Saved: {out}')
