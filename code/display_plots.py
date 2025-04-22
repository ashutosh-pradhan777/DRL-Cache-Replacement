import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

df = pd.read_csv("output/results.csv", header=None, names=[
    "trace_file", "trace_type", "algorithm", "hits", "misses", "writes", "filters",
    "cache_size", "cache_size_label", "cache_size_label_type", "hit_rate(%)",
    "avg_pollution", "time(s)", "compulsory", "capacity", "conflict"
])

# Optional: Clean and normalize
df["hit_rate(%)"] = pd.to_numeric(df["hit_rate(%)"], errors="coerce")
df["cache_size"] = pd.to_numeric(df["cache_size"], errors="coerce")

# Create output folder
output_dir = "plots/"
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x="algorithm", y="hit_rate(%)", hue="trace_type", ax=ax)
ax.set_title("Hit Rate per Algorithm per Trace")
ax.set_ylabel("Hit Rate (%)")
save_plot(fig, "hit_rate_per_algorithm_per_trace")

miss_df = df.groupby("algorithm")[["compulsory", "capacity", "conflict"]].sum()
fig, ax = plt.subplots(figsize=(10, 6))
miss_df.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Miss Breakdown per Algorithm")
ax.set_ylabel("Number of Misses")
save_plot(fig, "miss_breakdown_stacked_bar")

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df, x="cache_size", y="hit_rate(%)", hue="algorithm", marker="o", ax=ax)
ax.set_title("Hit Rate vs Cache Size")
save_plot(fig, "hit_rate_vs_cache_size")

algorithms = df["algorithm"].unique()
n = len(algorithms)
cols = 2
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
axes = axes.flatten()

for i, alg in enumerate(algorithms):
    sub = df[df["algorithm"] == alg]
    total = sub[["compulsory", "capacity", "conflict"]].sum()

    if total.sum() == 0:
        axes[i].text(0.5, 0.5, "No Misses", ha='center', va='center')
        axes[i].axis('off')
    else:
        axes[i].pie(total, labels=total.index, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f"Miss Type Breakdown - {alg.upper()}")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
save_plot(fig, "miss_type_pie_combined")


fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x="algorithm", y="time(s)", ax=ax)
ax.set_title("Execution Time per Algorithm")
ax.set_ylabel("Time (s)")
save_plot(fig, "execution_time_per_algorithm")
