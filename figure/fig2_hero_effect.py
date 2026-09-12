"""
figures/fig2_hero_effect.py

Figure 2: transition-specific orientation enrichment across
corpora (paper §5.2, Figure 2).

Values below are the matched-pseudo-anchor test results from
src/orientation_tests.py (see EXPECTED_HERO_TABLE there). This
script does not recompute them -- it only visualizes the reported
$\\Delta T$ values, so it can be regenerated instantly if any value
changes upstream.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif',
                      'mathtext.fontset': 'stix'})

# corpus, ΔT (pp), significant?, transition type
# -- keep in sync with orientation_tests.EXPECTED_HERO_TABLE --
ROWS = [
    ('Supreme Court',     3.0,  'sig', 'Procedural'),
    ('ANNO-MI',          -4.5,  'ns',  'Content'),
    ('ESConv',            9.5,  'ns',  'Content'),
    ('CMV',              22.6,  'sig', 'Content'),
    ('Korean counseling', 24.6, 'sig', 'Content'),
]

COLORS = {'sig': '#1E8A73', 'ns': '#B8B8B8'}
MARKERS = {'Procedural': 's', 'Content': 'o'}


def plot_hero_figure(rows=ROWS, out_path='fig2_final.png'):
    y_positions = [i * 1.6 for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(7, 5))
    for y, (name, dt, sig, transition_type) in zip(y_positions, rows):
        color = COLORS[sig]
        marker = MARKERS[transition_type]
        lw = 2.5 if sig == 'sig' else 1.3
        ms = 260 if sig == 'sig' else 160
        fs = 16 if sig == 'sig' else 14
        fc = '#222' if sig == 'sig' else '#888'
        fw = 'bold' if sig == 'sig' else 'normal'

        ax.plot([0, dt], [y, y], color=color, linewidth=lw, zorder=1)
        ax.scatter([dt], [y], color=color, marker=marker, s=ms,
                   edgecolors='white', linewidths=1.0, zorder=3)

        label = f"{dt:+.1f}pp" + (" (n.s.)" if sig == 'ns' else "")
        ax.annotate(label, xy=(dt, y), xytext=(0, 16),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=fs, fontweight=fw, color=fc)

    ax.axvline(0, color='#999', linewidth=0.9, linestyle='--', zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=15)
    ax.set_ylim(min(y_positions) - 1.1, max(y_positions) + 1.1)
    ax.set_xlabel(r'$\Delta T$ (percentage points, true anchor $-$ pseudo-anchor)',
                  fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)

    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['sig'],
               markersize=11, label='Content-defined transition'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['sig'],
               markersize=11, label='Procedural transition'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ns'],
               markersize=11, label='Not significant'),
    ]
    ax.legend(handles=legend_elems, loc='center right',
              bbox_to_anchor=(1.0, 0.14), fontsize=11, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig, ax


if __name__ == '__main__':
    plot_hero_figure()
    print("완료!")
