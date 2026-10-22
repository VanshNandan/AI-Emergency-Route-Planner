import matplotlib.pyplot as plt
import numpy as np

models = ["Decision Tree", "Random Forest"]

mae = [0.072494, 0.068319]
rmse = [0.097265, 0.090046]
r2 = [0.796690, 0.825750]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

metrics = [
    ("MAE", mae),
    ("RMSE", rmse),
    ("R²", r2)
]

for ax, (title, values) in zip(axes, metrics):
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.bar_label(bars, fmt="%.4f", padding=3)

fig.suptitle("Decision Tree vs Random Forest")
fig.tight_layout()
fig.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Saved model_comparison.png")