import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Example trajectories
turns = [1, 2, 3, 4, 5, 6, 7, 8]
jailbroken_scores = [1, 2, 3, 5, 6, 7, 9, 9]
benign_scores     = [1, 2, 1, 3, 2, 1, 2, 1]
threshold = 8

fig, ax = plt.subplots(figsize=(8, 4.5))

# Plot trajectories
ax.plot(turns, jailbroken_scores, marker='o', linewidth=2,
        color='#d62728', label='Jailbroken trajectory', zorder=3)
ax.plot(turns, benign_scores, marker='s', linewidth=2,
        color='#1f77b4', label='Benign trajectory', zorder=3)

# Threshold line
ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.2,
           label=f'Jailbreak threshold (score > {threshold})')

# Shade above threshold
ax.axhspan(threshold, 10.3, alpha=0.06, color='red')

# Annotate the crossing point
cross_turn = 7
cross_score = jailbroken_scores[cross_turn - 1]
ax.annotate('Threshold crossed\n(label: Jailbroken)',
            xy=(cross_turn, cross_score),
            xytext=(cross_turn - 2.2, cross_score + 0.6),
            fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='#d62728'),
            color='#d62728')

# Annotate benign max
ax.annotate('Max score = 3\n(label: Benign)',
            xy=(4, 3),
            xytext=(5.2, 4.5),
            fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='#1f77b4'),
            color='#1f77b4')

# Turn label annotations (show structure of a single turn)
ax.annotate('', xy=(3, -1.8), xytext=(2, -1.8),
            xycoords=('data', 'axes fraction'),
            textcoords=('data', 'axes fraction'),
            arrowprops=dict(arrowstyle='<->', color='gray'))
ax.text(2.5, -0.18, 'Turn $t$\n(prompt + response + score$_t$)',
        ha='center', va='top', fontsize=8, color='gray',
        transform=ax.get_xaxis_transform())

# Axes
ax.set_xlabel('Turn index $t$', fontsize=11)
ax.set_ylabel('Judge score', fontsize=11)
ax.set_xlim(0.5, 8.5)
ax.set_ylim(0, 10.3)
ax.set_xticks(turns)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
import os
save_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(save_dir, 'figure_trajectory.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(save_dir, 'figure_trajectory.png'), bbox_inches='tight', dpi=300)
print(f"Saved to: {save_dir}")
plt.show()
