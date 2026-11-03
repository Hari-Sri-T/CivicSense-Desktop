import matplotlib.pyplot as plt
import numpy as np
# Set style for academic paper
plt.style.use('ggplot')

# Data from tables
models = ['Qwen2.5-3B', 'Gemma-3-4B', 'Ministral-3B', 'Llama-3.2-3B', 'Nemotron-4B']

# BLEU Scores
phase1_bleu = [19.28, 16.39, 3.80, 0.00, 0.04]
phase2_bleu = [8.12, 19.94, 5.37, 3.95, 0.01]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, phase1_bleu, width, label='Phase 1: Unrestricted Reasoning', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, phase2_bleu, width, label='Phase 2: Restricted Reasoning', color='#ff7f0e', edgecolor='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Corpus BLEU Score', fontweight='bold')
ax.set_title('Impact of Reasoning Ablation on Generation Quality (BLEU)', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold', rotation=15)
ax.legend(fontsize=12, loc='upper right')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('ablation_bleu_chart.png', dpi=300, bbox_inches='tight')
plt.close()
