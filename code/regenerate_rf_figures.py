# -*- coding: utf-8 -*-
"""Regenerate RF figures (confusion matrix, F1 bar chart, feature importance)
from the leak-free RF results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "C:/Users/tosea/claude_test_project"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "nlp_results")

PERIOD_ORDER = [
    "1851-1900", "1901-1930", "1931-1950", "1951-1960",
    "1961-1980", "1981-20010910", "20010911-2019",
]
PERIOD_LABELS = [
    "1851-1900", "1901-1930", "1931-1950", "1951-1960",
    "1961-1980", "1981-2001", "2001-2019",
]

# ── Figure 3: Confusion Matrix Heatmap ─────────────────────────────
print("Generating Figure 3: Confusion Matrix...")
cm_df = pd.read_csv(os.path.join(RESULTS_DIR, "RF_Confusion_Matrix.csv"), index_col=0)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_df.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(PERIOD_LABELS)))
ax.set_yticks(range(len(PERIOD_LABELS)))
ax.set_xticklabels(PERIOD_LABELS, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(PERIOD_LABELS, fontsize=9)
ax.set_xlabel('Predicted Period', fontsize=11)
ax.set_ylabel('Actual Period', fontsize=11)
ax.set_title('Random Forest Confusion Matrix (Leak-Free, n=597/period)', fontsize=12)

for i in range(len(PERIOD_LABELS)):
    for j in range(len(PERIOD_LABELS)):
        val = cm_df.values[i, j]
        color = 'white' if val > cm_df.values.max() * 0.6 else 'black'
        ax.text(j, i, str(int(val)), ha='center', va='center', color=color, fontsize=10)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure4_RF_ConfusionMatrix.png"), dpi=200)
plt.close()
print("  Saved Figure4_RF_ConfusionMatrix.png")

# ── Figure 4: Per-Period F1 Bar Chart ──────────────────────────────
print("Generating Figure 4: Per-Period F1 Bar Chart...")
pc_df = pd.read_csv(os.path.join(RESULTS_DIR, "RF_Final_PerClass.csv"))
f1_scores = pc_df['F1-Score'].values
weighted_f1 = 0.666

colors = []
for f1 in f1_scores:
    if f1 >= 0.75:
        colors.append('#1a9876')  # dark teal
    elif f1 >= 0.65:
        colors.append('#4db89a')  # medium teal
    elif f1 >= 0.60:
        colors.append('#f0ad4e')  # orange
    else:
        colors.append('#d9534f')  # red

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(PERIOD_LABELS)), f1_scores, color=colors, edgecolor='gray', height=0.6)
ax.set_yticks(range(len(PERIOD_LABELS)))
ax.set_yticklabels(PERIOD_LABELS, fontsize=10)
ax.set_xlabel('F1-Score', fontsize=11)
ax.set_title('Random Forest F1-Score by Historical Period (Leak-Free)', fontsize=12)
ax.axvline(x=weighted_f1, color='navy', linestyle='--', linewidth=1.5,
           label=f'Weighted F1 = {weighted_f1:.3f}')
ax.set_xlim(0, 1.0)

for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
    ax.text(f1 + 0.02, i, f'{f1:.3f}', va='center', fontsize=9)

ax.legend(loc='lower right', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure5_RF_F1_ByPeriod.png"), dpi=200)
plt.close()
print("  Saved Figure5_RF_F1_ByPeriod.png")

# ── Figure 5: Feature Importance Bar Chart ─────────────────────────
print("Generating Figure 5: Feature Importance...")
fi_df = pd.read_csv(os.path.join(RESULTS_DIR, "RF_Feature_Importance.csv"),
                     index_col=0, header=0)
fi_df.columns = ['Importance']
top20 = fi_df.head(20).sort_values('Importance')

# Color coding
post911 = {'president_bush', 'iraq', 'bush_administration', 'sept_attack', 'afghanistan',
           'terrorist_attack', 'attack'}
artifacts = {'typhus'}

colors = []
for term in top20.index:
    if term in artifacts:
        colors.append('#999999')  # gray
    elif term in post911:
        colors.append('#d9534f')  # red
    else:
        colors.append('#337ab7')  # blue

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top20)), top20['Importance'].values, color=colors,
               edgecolor='gray', height=0.6)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.set_xlabel('Gini Importance', fontsize=11)
ax.set_title('Top-20 Random Forest Feature Importances (Leak-Free, 78,798 features)', fontsize=12)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d9534f', label='Post-9/11 vocabulary'),
    Patch(facecolor='#337ab7', label='Historical period vocabulary'),
    Patch(facecolor='#999999', label='OCR artifact'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_RF_FeatureImportance.png"), dpi=200)
plt.close()
print("  Saved Figure6_RF_FeatureImportance.png")

print("\nAll RF figures regenerated successfully.")
