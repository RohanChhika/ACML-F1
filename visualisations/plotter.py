import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Raw metric values
labels = ['RMSE (lower is better)', 'MAE (lower is better)', '±2s Accuracy', '±5s Accuracy']
model_names = ['v1', 'v2', 'v3']
metrics = {
    'v1': [7.85, 4.9, 30.4, 68.3],
    'v2': [6.3, 4.0, 52.0, 80.0],
    'v3': [5.24, 2.70, 59.9, 88.7]
}

# Normalize: invert error metrics so higher is better
metrics_np = np.array(list(metrics.values()))
normalized = []

for i in range(len(labels)):
    col = metrics_np[:, i]
    max_val = max(col)
    min_val = min(col)
    
    if 'lower is better' in labels[i]:
        norm_col = 1 - (col - min_val) / (max_val - min_val)
    else:
        norm_col = (col - min_val) / (max_val - min_val)
    
    normalized.append(norm_col)

# Transpose: model-wise stats
normalized_data = list(map(list, zip(*normalized)))
angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for model_name, stats in zip(model_names, normalized_data):
    stats += stats[:1]
    ax.plot(angles, stats, label=model_name)
    ax.fill(angles, stats, alpha=0.1)

# Format
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.set_title('Normalized Radar Chart: Higher = Better on All Axes', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.savefig("images/radar_chart_normalized_fixed_labels.png")
plt.show()
