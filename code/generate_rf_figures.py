# -*- coding: utf-8 -*-
"""
Generate three RF visualization figures for the manuscript:
  Figure 4 - Confusion Matrix Heatmap
  Figure 5 - Per-Period F1 Bar Chart
  Figure 6 - Top-20 Feature Importance Bar Chart
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUT_DIR = 'C:/Users/tosea/claude_test_project/'

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

PERIOD_SHORT = [
    '1851–1900',
    '1901–1930',
    '1931–1950',
    '1951–1960',
    '1961–1980',
    '1981–2001',
    '2001–2019',
]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Confusion Matrix Heatmap
# ══════════════════════════════════════════════════════════════════════════════
cm_data = [
    [107,  7,  0,  2,  1,  2,  0],
    [ 10, 80, 17,  8,  4,  0,  0],
    [  0, 20, 68, 25,  4,  3,  0],
    [  3, 10,  7, 97,  0,  2,  0],
    [  3, 10,  4, 16, 68, 15,  3],
    [  5,  3,  1,  5, 15, 67, 24],
    [  0,  1,  0,  0,  3, 17, 99],
]
cm = np.array(cm_data)

fig, ax = plt.subplots(figsize=(8.5, 7))

# Mask the diagonal so it gets a different color treatment
mask_diag = np.eye(7, dtype=bool)

# Off-diagonal heatmap (errors only) — light reds
off_diag = cm.copy().astype(float)
np.fill_diagonal(off_diag, np.nan)

# Correct-classification diagonal — light blues
diag_vals = np.full_like(cm, np.nan, dtype=float)
np.fill_diagonal(diag_vals, np.diag(cm))

# Draw off-diagonal first
sns.heatmap(off_diag, ax=ax, annot=True, fmt='.0f',
            cmap='Reds', linewidths=0.5, linecolor='white',
            cbar=False, annot_kws={'size': 10},
            xticklabels=PERIOD_SHORT, yticklabels=PERIOD_SHORT,
            vmin=0, vmax=18)

# Overlay diagonal in blue
sns.heatmap(diag_vals, ax=ax, annot=True, fmt='.0f',
            cmap='Blues', linewidths=0.5, linecolor='white',
            cbar=False, annot_kws={'size': 10, 'weight': 'bold'},
            xticklabels=False, yticklabels=False,
            vmin=60, vmax=115)

ax.set_xlabel('Predicted Period', labelpad=10)
ax.set_ylabel('Actual Period', labelpad=10)
ax.set_title('Figure 3. Random Forest Confusion Matrix\n'
             '(diagonal = correct classifications; off-diagonal = errors)',
             pad=12)
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)

# Legend patches
correct_patch = mpatches.Patch(facecolor='#3182bd', label='Correct (diagonal)')
error_patch   = mpatches.Patch(facecolor='#de2d26', label='Error (off-diagonal)')
ax.legend(handles=[correct_patch, error_patch],
          loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
path4 = OUT_DIR + 'Figure4_RF_Confusion_Heatmap.png'
fig.savefig(path4, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {path4}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-Period F1 Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
perf = pd.read_csv('C:/Users/tosea/claude_test_project/nlp_results/RF_Final_PerClass.csv')

# Use short labels in chronological order
period_map = {
    '1851-1900': '1851–1900',
    '1901-1930': '1901–1930',
    '1931-1950': '1931–1950',
    '1951-1960': '1951–1960',
    '1961-1980': '1961–1980',
    '1981-2001 (pre-9/11)': '1981–2001\n(pre-9/11)',
    '2001-2019 (post-9/11)': '2001–2019\n(post-9/11)',
}
perf['label'] = perf['Period'].map(period_map)

OVERALL_F1 = 0.697

# Color bars: darker teal for high F1, muted orange for low F1
f1_vals = perf['F1'].values
colors = []
for v in f1_vals:
    if v >= 0.80:
        colors.append('#1a6b8a')   # strong blue-teal
    elif v >= 0.72:
        colors.append('#4aabcc')   # mid teal
    elif v >= 0.68:
        colors.append('#f4a261')   # amber
    else:
        colors.append('#e76f51')   # orange-red (lowest)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(perf['label'], f1_vals, color=colors, edgecolor='white',
               linewidth=0.8, height=0.6)

# Value labels
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=10)

# Overall F1 reference line
ax.axvline(OVERALL_F1, color='#333333', linestyle='--', linewidth=1.4,
           label=f'Weighted avg. F1 = {OVERALL_F1}')

ax.set_xlabel('Weighted F1-Score')
ax.set_title('Figure 4. Random Forest Classification F1-Score by Historical Period\n'
             '(n = 597/period; balanced sampling with replacement; seed = 123)',
             pad=10)
ax.set_xlim(0, 0.93)
ax.invert_yaxis()   # earliest period at top
ax.legend(loc='lower right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
path5 = OUT_DIR + 'Figure5_RF_F1_ByPeriod.png'
fig.savefig(path5, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {path5}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Top-20 Feature Importance Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
fi = pd.read_csv('C:/Users/tosea/claude_test_project/nlp_results/RF_Feature_Importance.csv',
                 index_col=0)
fi.columns = ['Importance']
top20 = fi.head(20).copy()
top20.index = top20.index.str.replace('_', ' ')

# Categorise features
POST911 = {'terrorist', 'terrorism', 'president bush',
           'iraq', 'bush administration', 'terrorist attack', 'israeli',
           'israel', 'palestine'}
HISTORICAL = {'german', 'russia', 'jewish', 'algeria', 'communist',
              'bombing', 'russian', 'french', 'british', 'declare', 'spokesman'}
ARTIFACT   = set()   # no preprocessing artifacts in updated top-20

def classify(term):
    t = term.lower()
    if t in ARTIFACT:
        return 'artifact'
    if t in POST911:
        return 'post-9/11'
    if t in HISTORICAL:
        return 'historical'
    return 'historical'   # default for ambiguous (content terms)

CAT_COLORS = {
    'post-9/11':  '#c0392b',   # red
    'historical': '#2980b9',   # blue
    'artifact':   '#95a5a6',   # gray
}
top20['category'] = [classify(t) for t in top20.index]
top20['color']    = top20['category'].map(CAT_COLORS)
top20_rev = top20.iloc[::-1]   # plot lowest at bottom → rank 1 at top

fig, ax = plt.subplots(figsize=(8.5, 7))
bars = ax.barh(top20_rev.index, top20_rev['Importance'],
               color=top20_rev['color'], edgecolor='white',
               linewidth=0.6, height=0.7)

# Value labels
for bar, val in zip(bars, top20_rev['Importance']):
    ax.text(bar.get_width() + 0.00005, bar.get_y() + bar.get_height() / 2,
            f'{val:.5f}', va='center', ha='left', fontsize=8.5)

# Legend
legend_patches = [
    mpatches.Patch(color=CAT_COLORS['post-9/11'],  label='Post-9/11 vocabulary'),
    mpatches.Patch(color=CAT_COLORS['historical'],  label='Historical period vocabulary'),
    mpatches.Patch(color=CAT_COLORS['artifact'],    label='Preprocessing artifact / borderline'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)

ax.set_xlabel('Gini Feature Importance')
ax.set_title('Figure 5. Top-20 Random Forest Feature Importances\n'
             '(Gini criterion; CountVectorizer, 66,012 features)',
             pad=10)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0, top20['Importance'].max() * 1.22)

plt.tight_layout()
path6 = OUT_DIR + 'Figure6_RF_FeatureImportance.png'
fig.savefig(path6, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {path6}')

print('\nAll three figures generated successfully.')
