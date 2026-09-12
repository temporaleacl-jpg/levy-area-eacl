"""
figures/fig1_concept.py

Figure 1: Lévy-area geometry and the sign-invariant orientation
relation (paper §4.1, Figure 1).

Purely illustrative -- no data dependency. Panel (a) shows that two
paths with identical endpoints and length can have opposite Lévy
area sign due to directional ordering alone. Panel (b) shows the
orientation relation (same vs. different sign between a session's
baseline and pre-transition windows), which is invariant to the
arbitrary global sign of session-wise PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

plt.rcParams.update({'font.family': 'serif',
                      'mathtext.fontset': 'stix',
                      'font.size': 10})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# ══════════════════════════════════════════
# PANEL A -- Lévy area: same endpoints, different rotation
# ══════════════════════════════════════════
ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(-0.48, 0.42)

theta_a = np.linspace(np.pi, 0, 100)
x_a, y_a = 0.3 * np.cos(theta_a), 0.3 * np.sin(theta_a)
theta_b = np.linspace(np.pi, 2 * np.pi, 100)
x_b, y_b = 0.3 * np.cos(theta_b), 0.3 * np.sin(theta_b)

ax.fill_between(x_a, 0, y_a, where=(y_a >= 0), color='#E6F1FB', alpha=0.7)
ax.fill_between(x_b, y_b, 0, where=(y_b <= 0), color='#FAECE7', alpha=0.7)
ax.plot(x_a, y_a, color='#185FA5', linewidth=2.5, zorder=3)
ax.plot(x_b, y_b, color='#993C1D', linewidth=2.5, zorder=3)

for mid, xs, ys, col in [(50, x_a, y_a, '#185FA5'), (50, x_b, y_b, '#993C1D')]:
    ax.annotate('', xy=(xs[mid + 4], ys[mid + 4]), xytext=(xs[mid], ys[mid]),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.0, mutation_scale=14))

ax.scatter(-0.3, 0, c='#1D9E75', s=70, zorder=5, marker='o')
ax.scatter(0.3, 0, c='#444441', s=70, zorder=5, marker='s')
ax.text(-0.3, 0.05, 'start', ha='center', fontsize=8, color='#1D9E75')
ax.text(0.3, 0.05, 'end', ha='center', fontsize=8, color='#444441')
ax.text(0, 0.17, 'Lévy area < 0', ha='center', va='center',
        fontsize=9, color='#185FA5', fontweight='bold')
ax.text(0, -0.17, 'Lévy area > 0', ha='center', va='center',
        fontsize=9, color='#993C1D', fontweight='bold')
ax.text(0, -0.43, 'Same endpoints \u00b7 same path length\n'
        '\u2192 different directional ordering',
        ha='center', va='center', fontsize=8, color='#5F5E5A', style='italic')
ax.axhline(0, color='#B4B2A9', linewidth=0.8, linestyle='--', alpha=0.7)
ax.set_xlabel('semantic dim 1 (PCA)', fontsize=10)
ax.set_ylabel('semantic dim 2 (PCA)', fontsize=10)
ax.set_title('(a) Why Lévy area?\nSame endpoints, different rotation', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ══════════════════════════════════════════
# PANEL B -- orientation coupling definition (sign-invariant)
# ══════════════════════════════════════════
ax2 = axes[1]
ax2.set_xlim(0, 8.2)
ax2.set_ylim(0, 6.4)
ax2.axis('off')
ax2.set_title('(b) How orientation coupling is defined', fontsize=11, pad=10)


def draw_rotation_glyph(ax, cx, cy, direction='ccw', color='#222', r=0.55):
    if direction == 'ccw':
        theta1, theta2 = 25, 310
        arc = Arc((cx, cy), 2 * r, 2 * r, angle=0, theta1=theta1, theta2=theta2,
                  color=color, linewidth=3.0, zorder=3)
        ang = np.radians(theta2)
        tip = (cx + r * np.cos(ang), cy + r * np.sin(ang))
        tang = (-np.sin(ang), np.cos(ang))
    else:
        theta1, theta2 = 335, 50
        arc = Arc((cx, cy), 2 * r, 2 * r, angle=0, theta1=theta1, theta2=theta2,
                  color=color, linewidth=3.0, zorder=3)
        ang = np.radians(theta1)
        tip = (cx + r * np.cos(ang), cy + r * np.sin(ang))
        tang = (np.sin(ang), -np.cos(ang))
    ax.add_patch(arc)
    ax.annotate('', xy=(tip[0] + 0.15 * tang[0], tip[1] + 0.15 * tang[1]), xytext=tip,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=0, mutation_scale=16))


def draw_card(ax, y_center, dir1, dir2, accent_color, bg_color,
              glyph_color, label):
    box = FancyBboxPatch((0.2, y_center - 1.15), 7.8, 2.3,
                          boxstyle="round,pad=0.05,rounding_size=0.12",
                          facecolor=bg_color, edgecolor=accent_color,
                          linewidth=1.6, zorder=1)
    ax.add_patch(box)

    x_base, x_pre, x_label = 1.6, 4.0, 6.7

    ax.text(x_base, y_center + 0.75, 'Baseline', fontsize=9.5, ha='center',
            color='#333', style='italic', zorder=4)
    draw_rotation_glyph(ax, x_base, y_center - 0.15, direction=dir1, color=glyph_color)

    ax.text(x_pre, y_center + 0.75, 'Pre-transition', fontsize=9.5, ha='center',
            color='#333', style='italic', zorder=4)
    draw_rotation_glyph(ax, x_pre, y_center - 0.15, direction=dir2, color=glyph_color)

    arr = FancyArrowPatch((x_base + 0.75, y_center - 0.15), (x_pre - 0.75, y_center - 0.15),
                           arrowstyle='-|>', color='#888', lw=1.4,
                           mutation_scale=12, linestyle=(0, (3, 2)), zorder=2)
    ax.add_patch(arr)

    ax.text(x_label, y_center - 0.15, label, fontsize=11,
            color=accent_color, fontweight='bold', va='center', ha='center', zorder=4)


draw_card(ax2, 4.55, 'ccw', 'ccw',
          accent_color='#185FA5', bg_color='#DCEBFA',
          glyph_color='#185FA5', label='SAME\norientation')

draw_card(ax2, 1.95, 'ccw', 'cw',
          accent_color='#993C1D', bg_color='#FCE6DC',
          glyph_color='#993C1D', label='DIFFERENT\norientation')

ax2.text(4.1, 0.35,
         r'Global sign flip: $(\circlearrowleft,\circlearrowleft)'
         r' \equiv (\circlearrowright,\circlearrowright)$'
         '\nSame/different relation is invariant to arbitrary PCA axis sign flips.',
         fontsize=8.3, ha='center', color='#5F5E5A', style='italic')

plt.tight_layout()
plt.savefig('fig1_final.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("완료!")
