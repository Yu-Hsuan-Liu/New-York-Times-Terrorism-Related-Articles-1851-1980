# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# ── Data (percentages, from LDA_Topic_Proportions.csv; updated preprocessing) ──
periods = [
    '1851–1900', '1901–1930', '1931–1950',
    '1951–1960', '1961–1980',
    '1981–2001\n(pre-9/11)', '2001–2019\n(post-9/11)',
]

# Each list = proportions (%) across the 7 periods
topics = {
    'T1: European War & Rev. Politics':  [90.95, 66.81, 59.10, 44.52, 20.94,  3.32,  0.08],
    'T5: Cold War & Soviet Politics':    [ 7.20, 22.60, 15.37, 17.08, 11.56,  7.54,  3.68],
    'T0: Polit. Violence & Brit. Col.':  [ 0.17,  3.27,  6.08, 12.76, 17.30, 13.74,  2.81],
    'T9: Armed Violence Operations':     [ 0.00,  1.68,  7.52, 14.09, 13.46, 12.24,  6.74],
    'T10: Iraq & Afghanistan':           [ 0.00,  0.00,  0.07,  0.33,  2.17,  4.18, 17.41],
    'T8: Domestic & Social Violence':    [ 0.17,  0.62,  0.99,  0.73,  5.11, 11.91, 16.48],
    'T7: Israeli–Palestinian Conflict':  [ 0.00,  0.00,  1.87,  2.19,  9.02, 11.74,  8.33],
    'T6: Criminal Justice & Pros.':      [ 0.34,  3.62,  4.59,  3.19,  6.81, 11.64, 11.08],
    'T4: Electoral & Domestic Pol.':     [ 1.01,  0.53,  0.50,  0.53,  1.26,  3.29, 11.56],
    'Other (T2, T3, T11)':               [ 0.17,  0.88,  3.92,  4.59, 12.38, 20.41, 21.82],
}

# Colorblind-friendly palette (Tableau-10 inspired)
colors = [
    '#4e79a7',  # blue       – T5 European War
    '#59a14f',  # green      – T9 Anti-Colonial
    '#e15759',  # red        – T6 Criminal Violence
    '#f28e2b',  # orange     – T4 Electoral
    '#76b7b2',  # teal       – T2 Cold War
    '#9467bd',  # purple     – T7 Israeli-Palestinian
    '#d62728',  # dark red   – T8 S. Asia
    '#8c564b',  # brown      – T10 Iraq
    '#bcbd22',  # yellow-grn – T0 Governance
    '#c7c7c7',  # light grey – Other
]

x = np.arange(len(periods))
width = 0.62

fig, ax = plt.subplots(figsize=(13, 7))

bottom = np.zeros(len(periods))
for (label, vals), color in zip(topics.items(), colors):
    vals_arr = np.array(vals)
    ax.bar(x, vals_arr, width, bottom=bottom,
           label=label, color=color, edgecolor='white', linewidth=0.5)
    bottom += vals_arr

# Vertical dashed line marking 9/11 divide
ax.axvline(x=5.5, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(5.53, 97, '9/11', fontsize=9, color='#333333', va='top', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=9.5)
ax.set_ylabel('Proportion of Documents (%)', fontsize=11, labelpad=6)
ax.set_xlabel('Historical Period', fontsize=11, labelpad=8)
ax.set_ylim(0, 100)
ax.set_xlim(-0.45, len(periods) - 0.55)
ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.legend(
    loc='upper left', bbox_to_anchor=(1.01, 1.01),
    fontsize=9, framealpha=0.95, edgecolor='#cccccc',
    title='LDA Topics (K = 12)', title_fontsize=9.5,
    borderpad=0.8, labelspacing=0.5,
)

ax.set_title(
    'Figure 2. LDA Topic Proportions by Historical Period',
    fontsize=11, fontweight='bold', pad=10, loc='left',
)

plt.tight_layout()
out = 'C:/Users/tosea/claude_test_project/Figure2_LDA_Topics.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out)

# ── Heatmap (supplementary / alternative view) ──────────────────────────────
topic_labels_short = [
    'T1: Eur. War & Rev.', 'T5: Cold War', 'T0: Pol. Viol. & Brit.',
    'T9: Armed Viol.', 'T10: Iraq & Afghan.', 'T8: Dom. & Soc.',
    'T7: Isr.-Pal.', 'T6: Crim. Justice', 'T4: Electoral', 'Other',
]
heatmap_data = np.array(list(topics.values()))   # shape (10 topics, 7 periods)

period_labels_short = [
    '1851–\n1900', '1901–\n1930', '1931–\n1950', '1951–\n1960',
    '1961–\n1980', '1981–\n2001', '2001–\n2019',
]

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=80)
ax2.set_xticks(np.arange(len(periods)))
ax2.set_xticklabels(period_labels_short, fontsize=9.5)
ax2.set_yticks(np.arange(len(topic_labels_short)))
ax2.set_yticklabels(topic_labels_short, fontsize=9.5)

# Annotate cells
for i in range(len(topic_labels_short)):
    for j in range(len(periods)):
        val = heatmap_data[i, j]
        color = 'white' if val > 40 else 'black'
        ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                 fontsize=8, color=color)

cb = fig2.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
cb.set_label('% of Documents', fontsize=10)
ax2.set_title('Figure 2 (alt). LDA Topic Proportions Heatmap by Historical Period',
              fontsize=10, fontweight='bold', pad=10, loc='left')
ax2.set_xlabel('Historical Period', fontsize=10, labelpad=6)

plt.tight_layout()
out2 = 'C:/Users/tosea/claude_test_project/Figure2_LDA_Heatmap.png'
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out2)
