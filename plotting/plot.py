import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline


# -------------------------------------
# Plotting functions
# -------------------------------------

def plot_runtime_boxplot(data, method_name, set_instance_name):
    data = data.copy()
    data["n_str"] = data["n"].apply(lambda x: f"N{x}")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')

    custom_palette = sns.color_palette("bright")

    sns.boxplot(
        x="d",
        y="TotalTime",
        hue="n_str",
        data=data,
        palette=custom_palette,
        showfliers=True,
        flierprops=dict(marker='o', markerfacecolor='None', markersize=10,  markeredgecolor='white'),
        ax=ax
    )

    # Recolor box borders
    for artist in ax.artists:
        artist.set_edgecolor('white')
        artist.set_linewidth(1.5)

    # Set color of median lines, whiskers, caps
    for line in ax.lines:
        line.set_color('white')
        line.set_linewidth(1.0)

    num_categories = data["d"].nunique()
    num_hues = data["n_str"].nunique()
    expected_line_count = num_categories * num_hues * 6  # box components

    for line in ax.lines[expected_line_count:]:
        line.set_markerfacecolor('white')
        line.set_markeredgecolor('white')
        line.set_markersize(6)

    # Axis and title styling
    ax.set_xlabel("Arc density", fontsize=12, color='white')
    ax.set_ylabel("Time (s)", fontsize=12, color='white')
    ax.set_title(f"{method_name} Runtime by Arc Density and Node Count ({set_instance_name})", fontsize=14, color='white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Legend
    legend = ax.legend(title="n", loc="upper left", frameon=False)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Axes border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

def plot_matheuristic_relative_difference(df_mh, df_opt, set_instance_name):
    df_mh = df_mh.copy()
    df_opt = df_opt.copy()

    # Merge and compute Δ%
    df_merged = pd.merge(
        df_mh,
        df_opt,
        on=["n", "d", "instance_id"],
        suffixes=("_mh", "_opt")
    )

    # Compute average costs grouped by n and d
    avg_costs = df_merged.groupby(["d", "n"]).agg({
        "Cost_mh": "mean",
        "UB_opt": "mean"
    }).reset_index()

    # Compute Δ% = 100 * (MH - Optimal) / Optimal
    avg_costs["delta_percent"] = 100 * (avg_costs["Cost_mh"] - avg_costs["UB_opt"]) / avg_costs["UB_opt"]

    # Create string labels for n
    avg_costs["n_str"] = avg_costs["n"].apply(lambda x: f"N{x}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')

    custom_palette = sns.color_palette("bright", len(avg_costs["n_str"].unique()))

    d_categories = sorted(avg_costs["d"].unique())
    d_to_num = {cat: i for i, cat in enumerate(d_categories)}

    for (n_str, group), color in zip(avg_costs.groupby("n_str"), custom_palette):
        group = group.sort_values("d")
        x = group["d"].map(d_to_num).values
        y = group["delta_percent"].values

        # Smooth curve
        if len(x) >= 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=2)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=color, label=n_str)
        else:
            ax.plot(x, y, color=color, label=n_str)

        # Overlay original data points
        ax.scatter(x, y, color=color, edgecolor='white', s=60, zorder=3)

    # Axes and labels
    ax.set_xlabel("Arc density (d)", fontsize=12, color='white')
    ax.set_ylabel("Δ% (Relative Cost Difference)", fontsize=12, color='white')
    ax.set_title(f"Efficacy of Matheuristic vs Optimal ({set_instance_name})", fontsize=14, color='white')

    # Axes ticks and lines
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(list(d_to_num.values()))
    ax.set_xticklabels(list(d_to_num.keys()))

    # Legend
    legend = ax.legend(title="n", fontsize=10, loc="upper right", bbox_to_anchor=(0.95, 1), frameon=False)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Axes border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

def plot_matheuristic_runtime_bars(data):
    df_mh = data.copy()
    plt.rcParams.update({'axes.facecolor': 'white', 'figure.facecolor': 'white'})

    ### First plot: Node series

    # Average time grouped by arc density and n
    df_node = df_mh.groupby(['d', 'n'])['TotalTime'].mean().reset_index()
    df_node['n_str'] = df_node['n'].apply(lambda x: f"N{x}")
    df_node = df_node.sort_values(['d', 'n'])

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor('#050505')
    ax1.set_facecolor('#050505')

    custom_palette = sns.color_palette("bright")

    sns.barplot(
        data=df_node,
        x='d',
        y='TotalTime',
        hue='n_str',
        palette=custom_palette,
        ax=ax1
    )

    ax1.set_title("Node series", fontsize=13, color='white')
    ax1.set_xlabel("Arc density", color='white')
    ax1.set_ylabel("Time (s)", color='white')
    ax1.tick_params(colors='white')
    legend = ax1.legend(title='', loc='upper left', frameon=False)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

    ### Second plot: Conflict series

    def label_r(r):
        if r == 1:
            return 'CD1'
        elif r == 2:
            return 'CD2'
        else:
            return 'CD3'
    df_mh = data[data['n'].isin([300, 400, 500])].copy()
    df_mh['r_value'] = df_mh['instance_id'].str.extract(r'R(\d*\.?\d+)').astype(float)
    df_mh['r'] = df_mh['r_value'].apply(label_r)
    df_conflict = df_mh.groupby(['d', 'r'])['TotalTime'].mean().reset_index()
    df_conflict = df_conflict.sort_values(['d', 'r'])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#050505')
    ax2.set_facecolor('#050505')

    sns.barplot(
        data=df_conflict,
        x='d',
        y='TotalTime',
        hue='r',
        palette=custom_palette,
        ax=ax2
    )

    ax2.set_title("Conflict series", fontsize=13, color='white')
    ax2.set_xlabel("Arc density", color='white')
    ax2.set_ylabel("Time (s)", color='white')
    ax2.tick_params(colors='white')
    legend = ax2.legend(title='', loc='upper left', frameon=False)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# Plotting ILP, MILP, Matheuristic (RAND and SWN)
# ------------------------------------------------

if __name__ == "__main__":
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Read the RND_experimental_results.csv file
    rnd_df = pd.read_csv("RND_experimental_results_modified.csv")
    plot_runtime_boxplot(rnd_df[rnd_df["method"] == "ILP"], "ILP", "RAND") # Plot for ILP (RAND)
    plot_runtime_boxplot(rnd_df[rnd_df["method"] == "MILP"], "MILP", "RAND") # Plot for MILP (RAND)
    plot_matheuristic_relative_difference(rnd_df[rnd_df["method"] == "Matheuristic"], rnd_df[rnd_df["method"] == "MILP"], "RAND") # Plot for Matheuristic (RAND)
    plot_matheuristic_runtime_bars(rnd_df[rnd_df["method"] == "Matheuristic"])

    # Read the SWN_experimental_results.csv file
    swn_df = pd.read_csv("SWN_experimental_results.csv")
    plot_runtime_boxplot(swn_df[swn_df["method"] == "ILP"], "ILP", "SWN") # Plot for ILP (SWN)
    plot_runtime_boxplot(swn_df[swn_df["method"] == "MILP"], "MILP", "SWN") # Plot for MILP (SWN)
    plot_matheuristic_relative_difference(swn_df[swn_df["method"] == "Matheuristic"], swn_df[swn_df["method"] == "MILP"], "SWN") # Plot for Matheuristic (SWN)