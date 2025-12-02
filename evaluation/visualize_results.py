"""
Visualize evaluation results from all three methods

Creates various charts and graphs to compare recommendation algorithms
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_results():
    """Load all evaluation results"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "eval_results")

    # Load CSV files - 3 evaluation methods:
    # 1. Centroid method (echo_chamber_analysis.csv)
    # 2. LLM evaluation (llm_evaluation.csv)
    # 3. Diversity method (echo_chamber_analysis.csv - same file, different metrics)
    centroid_df = pd.read_csv(os.path.join(results_dir, "echo_chamber_analysis.csv"))
    llm_df = pd.read_csv(os.path.join(results_dir, "llm_evaluation.csv"))
    diversity_df = pd.read_csv(os.path.join(results_dir, "echo_chamber_analysis.csv"))

    return centroid_df, llm_df, diversity_df

def plot_centroid_results(df, save_path="eval_results/centroid_visualizations.png"):
    """Visualize embedding drift by plotting actual centroid locations in embedding space"""
    import json
    import sys

    # Add path for imports
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        from recommendation_algorithms.news_corpus import NewsCorpus
        from recommendation_algorithms.full_run import load_mind_as_articles
        from evaluation.centroid_method import compute_centroid
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        print("Creating simplified visualization...")
        return plot_centroid_results_simple(df, save_path)

    # Load data - try multiple paths
    print("Loading embeddings and recommendation data...")
    MIND_PATHS = [
        "/content/MINDsmall_train",  # Colab
        os.path.join(base_dir, "MINDsmall_train"),  # Local
        os.path.join(os.path.dirname(base_dir), "MINDsmall_train")  # Parent dir
    ]

    MIND_PATH = None
    for path in MIND_PATHS:
        if os.path.exists(path):
            MIND_PATH = path
            break

    if MIND_PATH is None:
        print("Warning: MIND dataset not found. Creating simplified visualization...")
        return plot_centroid_results_simple(df, save_path)

    try:
        articles_df, behaviors, news_id_to_index = load_mind_as_articles(MIND_PATH)
        corpus = NewsCorpus(articles_df)

        # Try to load embeddings if they exist, otherwise create them
        embeddings_path = os.path.join(base_dir, "embeddings.npy")
        if os.path.exists(embeddings_path):
            print("Loading pre-computed embeddings...")
            corpus.load_embeddings(embeddings_path)
        else:
            print("Creating embeddings (this may take a while)...")
            corpus.create_embeddings()
            # Save for future use
            corpus.save_embeddings(embeddings_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating simplified visualization...")
        return plot_centroid_results_simple(df, save_path)

    # Load recommendation files - try multiple locations
    json_locations = [
        "/content",  # Colab
        os.path.join(base_dir, "recommended_results"),  # Local recommended_results folder
        base_dir  # Root directory
    ]

    recommendation_files = {
        "Pure Relevance": None,
        "Calibrated Diversity": None,
        "Serendipity-Aware": None
    }

    # Find JSON files
    for method, filename in [("Pure Relevance", "pure_relevance.json"),
                             ("Calibrated Diversity", "calibrated_diversity.json"),
                             ("Serendipity-Aware", "serendipity.json")]:
        for loc in json_locations:
            file_path = os.path.join(loc, filename)
            if os.path.exists(file_path):
                recommendation_files[method] = file_path
                break

    # Sample users for visualization (use first 200 for better visualization)
    sample_size = 200

    # Compute centroids for each method
    history_centroids_all = []
    rec_centroids = {}
    method_history_centroids_dict = {}  # Store history centroids per method
    method_colors = {'Pure Relevance': '#FF6B6B', 'Calibrated Diversity': '#4ECDC4', 'Serendipity-Aware': '#95E1D3'}

    # Build user history map
    print("Building user history map...")
    user_history_map = {}
    for _, row in behaviors.iterrows():
        if pd.isna(row["history"]):
            continue
        history_ids = row["history"].split()
        user_history_indices = [news_id_to_index[nid] for nid in history_ids if nid in news_id_to_index]
        if user_history_indices:
            user_history_map[str(row["user_id"])] = user_history_indices

    print(f"Found {len(user_history_map)} users with history")

    # Compute centroids for each recommendation method
    # IMPORTANT: All methods use the same users, so we should use the SAME history centroids
    # But we'll compute them once and reuse for all methods
    shared_history_centroids = {}
    method_drifts = {}

    for method_name, file_path in recommendation_files.items():
        if file_path is None or not os.path.exists(file_path):
            print(f"Warning: {method_name} JSON file not found, skipping...")
            continue

        print(f"Processing {method_name}...")
        with open(file_path, 'r') as f:
            recommendations = json.load(f)

        rec_centroids[method_name] = []
        method_history_centroids = []
        user_drifts = []

        count = 0
        for rec in recommendations:
            if count >= sample_size:
                break
            user_id = rec["user_id"]
            if user_id not in user_history_map:
                continue

            history_indices = user_history_map[user_id]
            recommended_indices = rec["recommended_indices"]

            try:
                # Compute centroids from actual embeddings
                # Use shared history centroids (same for all methods since same users)
                if user_id not in shared_history_centroids:
                    shared_history_centroids[user_id] = compute_centroid(corpus.embeddings, history_indices)

                hist_centroid = shared_history_centroids[user_id]
                rec_centroid = compute_centroid(corpus.embeddings, recommended_indices)

                method_history_centroids.append(hist_centroid)
                rec_centroids[method_name].append(rec_centroid)

                # Compute actual drift for this user
                from evaluation.centroid_method import compute_cosine_similarity
                cos_sim = compute_cosine_similarity(hist_centroid, rec_centroid)
                drift = 1 - cos_sim
                user_drifts.append(drift)

                count += 1
            except Exception as e:
                print(f"Error computing centroids for user {user_id}: {e}")
                continue

        if method_history_centroids:
            # Store method-specific history centroids (but they're actually the same across methods)
            method_history_centroids_dict[method_name] = method_history_centroids
            history_centroids_all.extend(method_history_centroids)
            method_drifts[method_name] = user_drifts
            avg_drift = np.mean(user_drifts) if user_drifts else 0
            print(f"  Computed {count} centroids for {method_name}, avg drift = {avg_drift:.4f}")

    if not rec_centroids or all(len(v) == 0 for v in rec_centroids.values()):
        print("No recommendation data found. Creating simplified visualization...")
        return plot_centroid_results_simple(df, save_path)

    # Reduce to 2D using PCA
    from sklearn.decomposition import PCA
    print("Reducing dimensions for visualization...")

    # Combine all centroids for PCA fitting
    all_centroids_list = []
    if history_centroids_all:
        all_centroids_list.append(np.vstack(history_centroids_all))
    for method_centroids in rec_centroids.values():
        if method_centroids:
            all_centroids_list.append(np.vstack(method_centroids))

    if not all_centroids_list:
        print("No centroids computed. Creating simplified visualization...")
        return plot_centroid_results_simple(df, save_path)

    all_centroids = np.vstack(all_centroids_list)

    pca = PCA(n_components=2)
    pca.fit(all_centroids)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Transform centroids - use all history centroids
    if history_centroids_all:
        history_2d = pca.transform(np.vstack(history_centroids_all))
    else:
        # Fallback: use first method's history centroids
        first_method = list(rec_centroids.keys())[0]
        if rec_centroids[first_method]:
            history_2d = pca.transform(np.vstack(rec_centroids[first_method]))
        else:
            return plot_centroid_results_simple(df, save_path)

    # Create visualization with 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: All centroids in 2D space
    ax = axes[0]

    # Plot history centroids (gray circles) - what users have read
    ax.scatter(history_2d[:, 0], history_2d[:, 1], c='gray', alpha=0.3, s=15,
               label='Circles = History (what users read)', marker='o')

    # Plot recommendation centroids for each method (colored triangles) - what algorithms recommend
    for method_name, method_centroids in rec_centroids.items():
        if method_centroids:
            rec_2d = pca.transform(np.vstack(method_centroids))
            color = method_colors.get(method_name, '#000000')
            ax.scatter(rec_2d[:, 0], rec_2d[:, 1], c=color, alpha=0.5, s=25,
                      label=f'Triangles = {method_name} (recommended)', marker='^', edgecolors='black', linewidths=0.3)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Centroid Locations in Embedding Space\n(Circles = History, Triangles = Recommendations)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Drift visualization based on ACTUAL drift values (not PCA distances)
    ax = axes[1]

    # Collect drift values for all methods
    methods_with_drift = []
    for method_name, method_centroids in rec_centroids.items():
        if method_centroids and method_name in method_drifts:
            avg_user_drift = np.mean(method_drifts[method_name])
            methods_with_drift.append((method_name, avg_user_drift))
            print(f"  {method_name}: Average per-user drift = {avg_user_drift:.4f} (lower = closer to history)")

    # Sort by drift (lowest first = closest to history)
    methods_with_drift.sort(key=lambda x: x[1])

    if not methods_with_drift:
        ax.text(0.5, 0.5, 'No drift data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Drift Visualization', fontsize=14, fontweight='bold')
        return

    # Create a radial/linear visualization where distance from center = drift value
    # History is at center (0, 0), methods are positioned at distance = drift
    history_pos = (0, 0)

    # Plot history at center
    ax.scatter(history_pos[0], history_pos[1], c='gray', s=200,
              marker='o', label='History (what users read)', edgecolors='black', linewidths=2, zorder=10)
    ax.annotate('History\n(center)', xy=history_pos, xytext=(0, 15), textcoords='offset points',
               fontsize=10, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))

    # Position methods in a circle around history, with radius = drift value
    # Use angles to spread them out visually
    max_drift = max(drift for _, drift in methods_with_drift)
    angles = np.linspace(0, 2*np.pi, len(methods_with_drift), endpoint=False)

    for i, (method_name, drift) in enumerate(methods_with_drift):
        color = method_colors.get(method_name, '#000000')

        # Position at angle, with radius proportional to drift
        # Scale drift to make visualization clearer (multiply by factor for visibility)
        scale_factor = 5.0  # Makes differences more visible
        radius = drift * scale_factor
        angle = angles[i]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Plot recommendation method
        ax.scatter(x, y, c=color, s=200,
                  marker='^', label=f'{method_name}', edgecolors='black', linewidths=2, zorder=10)
        ax.annotate(method_name, xy=(x, y), xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.4, edgecolor='black', linewidth=1.5))

        # Draw line from history to method (length = actual drift)
        ax.plot([history_pos[0], x], [history_pos[1], y],
               color=color, alpha=0.7, linewidth=2, linestyle='--', zorder=4)
        # Arrow at the end
        ax.annotate('', xy=(x, y), xytext=(x - (x-history_pos[0])*0.1, y - (y-history_pos[1])*0.1),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.7))

        # Add drift value
        ax.text(x, y - radius*0.15, f'drift={drift:.3f}',
               ha='center', va='top', fontsize=9, color=color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5))

    # Set equal aspect and limits
    max_radius = max_drift * scale_factor * 1.2
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect('equal')
    ax.set_xlabel('Distance from History (scaled by drift)', fontsize=11)
    ax.set_ylabel('Distance from History (scaled by drift)', fontsize=11)
    ax.set_title('Drift Visualization: Distance = Actual Drift Value\n(Closer to center = closer to history)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Bar chart showing the actual metric values
    ax = axes[2]

    # Get metric values from the dataframe
    methods_list = df['Method'].tolist()
    echo_scores = df['Echo_Score'].astype(float).tolist()
    rec_diversity = df['Rec_Div'].astype(float).tolist()

    x = np.arange(len(methods_list))
    width = 0.35

    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, echo_scores, width, label='Echo Score (lower is better)',
                   color='#FF6B6B', alpha=0.7)
    bars2 = ax.bar(x + width/2, rec_diversity, width, label='Recommendation Diversity',
                   color='#4ECDC4', alpha=0.7)

    # Add value labels on bars
    for i, (echo, div) in enumerate(zip(echo_scores, rec_diversity)):
        ax.text(i - width/2, echo + 0.01, f'{echo:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i + width/2, div + 0.01, f'{div:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Centroid Method Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_list, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_centroid_individual(df, output_dir):
    """Create individual plots for centroid method"""
    # Echo Score plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(df['Method'], df['Echo_Score'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Echo Chamber Score (lower is better)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Echo Score', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Echo_Score']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "echo_score.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Recommendation Diversity plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(df['Method'], df['Rec_Div'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Recommendation Diversity', fontsize=16, fontweight='bold')
    ax.set_ylabel('Diversity Score', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Rec_Div']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recommendation_diversity.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved individual centroid plots to {output_dir}/")

def plot_llm_individual(df, output_dir):
    """Create individual plots for LLM method"""
    import json

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "eval_results", "llm_reasoning")

    detailed_data = {}
    json_files = {
        "Pure Relevance": "llm_eval_pure_relevance.json",
        "Calibrated Diversity": "llm_eval_calibrated_diversity.json",
        "Serendipity-Aware": "llm_eval_serendipity-aware.json"
    }

    for method_name, filename in json_files.items():
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    detailed_data[method_name] = json.load(f)
            except:
                pass

    # Novelty distribution plot
    if detailed_data:
        novelty_data = []
        method_labels = []
        for method_name in df['Method']:
            data_key = method_name
            if data_key not in detailed_data:
                if "Serendipity" in method_name:
                    data_key = "Serendipity-Aware"
                elif "Diversity" in method_name:
                    data_key = "Calibrated Diversity"
                elif "Relevance" in method_name:
                    data_key = "Pure Relevance"

            if data_key in detailed_data:
                method_labels.append(method_name)
                novelty_data.append([float(item['novelty']) for item in detailed_data[data_key]])

        if novelty_data:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            parts = ax.violinplot(novelty_data, positions=range(len(method_labels)),
                                 widths=0.7, showmeans=True, showmedians=True)
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(method_labels)))
            ax.set_xticklabels(method_labels, rotation=15)
            ax.set_ylabel('Novelty Score', fontsize=14)
            ax.set_title('Novelty Score Distribution (1-5 scale)', fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 5.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "novelty_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # Perspective distribution plot
    if detailed_data:
        perspective_data = []
        method_labels = []
        for method_name in df['Method']:
            data_key = method_name
            if data_key not in detailed_data:
                if "Serendipity" in method_name:
                    data_key = "Serendipity-Aware"
                elif "Diversity" in method_name:
                    data_key = "Calibrated Diversity"
                elif "Relevance" in method_name:
                    data_key = "Pure Relevance"

            if data_key in detailed_data:
                method_labels.append(method_name)
                perspective_data.append([float(item['perspective']) for item in detailed_data[data_key]])

        if perspective_data:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            parts = ax.violinplot(perspective_data, positions=range(len(method_labels)),
                                 widths=0.7, showmeans=True, showmedians=True)
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(method_labels)))
            ax.set_xticklabels(method_labels, rotation=15)
            ax.set_ylabel('Perspective Score', fontsize=14)
            ax.set_title('Perspective Contrast Distribution (1-5 scale)', fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 5.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "perspective_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"  Saved individual LLM plots to {output_dir}/")

def plot_diversity_individual(df, output_dir):
    """Create individual plots for diversity method"""
    # Echo Chamber Percentage
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if df['Echo_Chamber_%'].dtype == 'object':
        echo_pct = df['Echo_Chamber_%'].str.rstrip('%').astype(float)
    else:
        echo_pct = df['Echo_Chamber_%'].astype(float)
    ax.bar(df['Method'], echo_pct, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Echo Chamber Percentage', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(echo_pct):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "echo_chamber_percentage.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Bubble Breaking Percentage
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if df['Bubble_Breaking_%'].dtype == 'object':
        bubble_pct = df['Bubble_Breaking_%'].str.rstrip('%').astype(float)
    else:
        bubble_pct = df['Bubble_Breaking_%'].astype(float)
    ax.bar(df['Method'], bubble_pct, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Bubble Breaking Percentage', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(bubble_pct):
        ax.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bubble_breaking_percentage.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved individual diversity plots to {output_dir}/")

def plot_centroid_results_simple(df, save_path="eval_results/centroid_visualizations.png"):
    """Fallback simple bar chart visualization using echo_chamber_analysis.csv"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Echo Score (lower is better)
    axes[0].bar(df['Method'], df['Echo_Score'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    axes[0].set_title('Echo Chamber Score (lower is better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Echo Score', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Echo_Score']):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Recommendation Diversity
    axes[1].bar(df['Method'], df['Rec_Div'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    axes[1].set_title('Recommendation Diversity', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Diversity Score', fontsize=12)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Rec_Div']):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_llm_results(df, save_path="eval_results/llm_visualizations.png"):
    """Visualize LLM evaluation results with box plots"""
    import json

    # Try to load detailed JSON files for box plots
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "eval_results", "llm_reasoning")

    # Load detailed data if available
    detailed_data = {}
    json_files = {
        "Pure Relevance": "llm_eval_pure_relevance.json",
        "Calibrated Diversity": "llm_eval_calibrated_diversity.json",
        "Serendipity-Aware": "llm_eval_serendipity-aware.json"
    }

    for method_name, filename in json_files.items():
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    detailed_data[method_name] = json.load(f)
                print(f"Loaded {len(detailed_data[method_name])} evaluations for {method_name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    print(f"Total methods loaded: {len(detailed_data)}")

    # Clean percentage column
    if df['Framing_Diff_%'].dtype == 'object':
        df['Framing_Diff'] = df['Framing_Diff_%'].str.rstrip('%').astype(float)
    else:
        df['Framing_Diff'] = df['Framing_Diff_%'].astype(float)
    df['Novelty'] = df['Novelty'].astype(float)
    df['Perspective'] = df['Perspective'].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data for box plots
    if detailed_data:
        novelty_data = []
        perspective_data = []
        framing_data = []
        method_labels = []

        # Match methods from CSV to JSON data
        for method_name in df['Method']:
            # Try exact match first
            data_key = method_name
            if data_key not in detailed_data:
                # Try alternative names
                if "Serendipity" in method_name:
                    data_key = "Serendipity-Aware"
                elif "Diversity" in method_name:
                    data_key = "Calibrated Diversity"
                elif "Relevance" in method_name:
                    data_key = "Pure Relevance"

            if data_key in detailed_data and len(detailed_data[data_key]) > 0:
                method_labels.append(method_name)
                novelty_data.append([float(item['novelty']) for item in detailed_data[data_key]])
                perspective_data.append([float(item['perspective']) for item in detailed_data[data_key]])
                framing_data.append([int(item['framing_different']) for item in detailed_data[data_key]])
                print(f"Added data for {method_name}: {len(detailed_data[data_key])} items")

        # Violin plot for Novelty (alternative to box plot)
        if novelty_data and len(novelty_data) > 0:
            print(f"Creating Novelty violin plot with {len(novelty_data)} methods")
            parts1 = axes[0, 0].violinplot(novelty_data, positions=range(len(method_labels)),
                                          widths=0.7, showmeans=True, showmedians=True)
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
            for i, pc in enumerate(parts1['bodies']):
                color = colors[i % len(colors)]
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            axes[0, 0].set_xticks(range(len(method_labels)))
            axes[0, 0].set_xticklabels(method_labels, rotation=15)
            axes[0, 0].set_ylabel('Novelty Score', fontsize=12)
            axes[0, 0].set_title('Novelty Score Distribution (1-5 scale)', fontsize=14, fontweight='bold')
            axes[0, 0].grid(axis='y', alpha=0.3)
            axes[0, 0].set_ylim(0, 5.5)
        else:
            print("Warning: No novelty data for plot")
            axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Novelty Score Distribution', fontsize=14, fontweight='bold')

        # Violin plot for Perspective (alternative to box plot)
        if perspective_data and len(perspective_data) > 0:
            print(f"Creating Perspective violin plot with {len(perspective_data)} methods")
            parts2 = axes[0, 1].violinplot(perspective_data, positions=range(len(method_labels)),
                                          widths=0.7, showmeans=True, showmedians=True)
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
            for i, pc in enumerate(parts2['bodies']):
                color = colors[i % len(colors)]
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            axes[0, 1].set_xticks(range(len(method_labels)))
            axes[0, 1].set_xticklabels(method_labels, rotation=15)
            axes[0, 1].set_ylabel('Perspective Score', fontsize=12)
            axes[0, 1].set_title('Perspective Contrast Distribution (1-5 scale)', fontsize=14, fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)
            axes[0, 1].set_ylim(0, 5.5)
        else:
            print("Warning: No perspective data for plot")
            axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Perspective Contrast Distribution', fontsize=14, fontweight='bold')

        # Framing Difference (bar chart with percentages)
        framing_percentages = [np.mean(f) * 100 for f in framing_data]
        axes[1, 0].bar(method_labels, framing_percentages, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 0].set_title('Political Framing Difference (%)', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=15)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim(0, 100)

        for i, v in enumerate(framing_percentages):
            axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Combined comparison - violin plots
        data_for_violin = []
        labels_for_violin = []
        for method_name in method_labels:
            data_for_violin.append(novelty_data[method_labels.index(method_name)])
            labels_for_violin.append(f'{method_name}\nNovelty')
            data_for_violin.append(perspective_data[method_labels.index(method_name)])
            labels_for_violin.append(f'{method_name}\nPerspective')

        parts = axes[1, 1].violinplot(data_for_violin, positions=range(len(data_for_violin)),
                                      widths=0.7, showmeans=True, showmedians=True)
        colors_cycle = ['#FF6B6B', '#4ECDC4', '#95E1D3'] * 2
        for pc, color in zip(parts['bodies'], colors_cycle):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        axes[1, 1].set_xticks(range(len(labels_for_violin)))
        axes[1, 1].set_xticklabels(labels_for_violin, rotation=15, ha='right')
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Combined Distribution Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim(0, 5.5)

    else:
        # Fallback to simple bar charts if no detailed data
        x = np.arange(len(df['Method']))
        width = 0.6

        axes[0, 0].bar(x, df['Novelty'], width, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(df['Method'], rotation=15)
        axes[0, 0].set_ylabel('Novelty Score', fontsize=12)
        axes[0, 0].set_title('Novelty Scores (1-5 scale)', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)

        axes[0, 1].bar(x, df['Perspective'], width, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['Method'], rotation=15)
        axes[0, 1].set_ylabel('Perspective Score', fontsize=12)
        axes[0, 1].set_title('Perspective Contrast (1-5 scale)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)

        axes[1, 0].bar(df['Method'], df['Framing_Diff'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
        axes[1, 0].set_title('Political Framing Difference (%)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Percentage', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=15)
        axes[1, 0].grid(axis='y', alpha=0.3)

        for i, v in enumerate(df['Framing_Diff']):
            axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_diversity_results(df, save_path="eval_results/diversity_visualizations.png"):
    """Visualize diversity analysis results (from echo_chamber_analysis.csv)"""
    # Clean numeric columns
    numeric_cols = ['Echo_Score', 'History_Div', 'Rec_Div',
                     'Echo_Chamber_%', 'Bubble_Breaking_%']
    for col in numeric_cols:
        if col in df.columns:
            if '%' in col:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.rstrip('%').astype(float)
                else:
                    df[col] = df[col].astype(float)
            else:
                df[col] = df[col].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Echo Chamber Score (no std dev bars)
    axes[0, 0].bar(df['Method'], df['Echo_Score'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    axes[0, 0].set_xlabel('Method', fontsize=12)
    axes[0, 0].set_ylabel('Echo Score', fontsize=12)
    axes[0, 0].set_title('Echo Chamber Score (lower is better)', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['Echo_Score']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Diversity comparison (no std dev bars)
    x = np.arange(len(df['Method']))
    width = 0.35
    axes[0, 1].bar(x - width/2, df['History_Div'], width, label='History Diversity', color='#4ECDC4', alpha=0.8)
    axes[0, 1].bar(x + width/2, df['Rec_Div'], width, label='Recommendation Diversity', color='#95E1D3', alpha=0.8)
    axes[0, 1].set_xlabel('Method', fontsize=12)
    axes[0, 1].set_ylabel('Diversity Score', fontsize=12)
    axes[0, 1].set_title('Diversity: History vs Recommendations', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df['Method'], rotation=15)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Echo Chamber Percentage
    axes[1, 0].bar(df['Method'], df['Echo_Chamber_%'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    axes[1, 0].set_title('Echo Chamber Percentage', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=15)
    axes[1, 0].grid(axis='y', alpha=0.3)

    for i, v in enumerate(df['Echo_Chamber_%']):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Bubble Breaking Percentage
    axes[1, 1].bar(df['Method'], df['Bubble_Breaking_%'], color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    axes[1, 1].set_title('Bubble Breaking Percentage', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(df['Bubble_Breaking_%']):
        axes[1, 1].text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def create_combined_table_and_visualization(centroid_df, llm_df, diversity_df, save_path="eval_results/combined_comparison.png"):
    """Create comprehensive table and visualization comparing all three evaluation methods"""

    # Create comprehensive comparison table
    methods = centroid_df['Method'].tolist()

    # Prepare data for table
    table_data = []
    for method in methods:
        # Centroid method metrics
        centroid_row = centroid_df[centroid_df['Method'] == method].iloc[0]
        echo_score = float(centroid_row['Echo_Score'])
        rec_div = float(centroid_row['Rec_Div'])
        echo_pct_val = centroid_row['Echo_Chamber_%']
        if isinstance(echo_pct_val, str):
            echo_pct = float(echo_pct_val.rstrip('%'))
        else:
            echo_pct = float(echo_pct_val)

        # LLM metrics
        llm_row = llm_df[llm_df['Method'] == method].iloc[0]
        novelty = float(llm_row['Novelty'])
        perspective = float(llm_row['Perspective'])
        framing_val = llm_row['Framing_Diff_%']
        if isinstance(framing_val, str):
            framing = float(framing_val.rstrip('%'))
        else:
            framing = float(framing_val)

        table_data.append({
            'Method': method,
            'Echo Score (lower better)': f'{echo_score:.3f}',
            'Rec Diversity': f'{rec_div:.3f}',
            'Echo Chamber %': f'{echo_pct:.1f}%',
            'LLM Novelty (1-5)': f'{novelty:.2f}',
            'LLM Perspective (1-5)': f'{perspective:.2f}',
            'Framing Diff %': f'{framing:.1f}%'
        })

    # Create cleaner visualization - organized by evaluation method
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Table (top, spans all columns)
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('tight')
    ax_table.axis('off')

    table_df = pd.DataFrame(table_data)
    table = ax_table.table(cellText=table_df.values, colLabels=table_df.columns,
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style the table
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_df) + 1):
        colors = ['#FFE5E5', '#E5F5F5', '#E5F5E5']
        for j in range(len(table_df.columns)):
            table[(i, j)].set_facecolor(colors[(i-1) % len(colors)])

    ax_table.set_title('Comprehensive Evaluation Results - All Three Methods',
                       fontsize=16, fontweight='bold', pad=20)

    # Chart 1: Centroid Method Metrics (Echo Score & Rec Diversity)
    ax1 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(methods))
    width = 0.35
    echo_scores = [float(centroid_df[centroid_df['Method'] == m]['Echo_Score'].iloc[0]) for m in methods]
    rec_divs = [float(centroid_df[centroid_df['Method'] == m]['Rec_Div'].iloc[0]) for m in methods]

    bars1 = ax1.bar(x - width/2, echo_scores, width, label='Echo Score\n(lower is better)',
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, rec_divs, width, label='Rec Diversity\n(higher is better)',
                    color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{echo_scores[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f'{rec_divs[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Recommendation Method', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('1. Centroid Method Metrics', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Chart 2: LLM Method Metrics (Novelty & Perspective)
    ax2 = fig.add_subplot(gs[1, 1])
    novelty_vals = [float(llm_df[llm_df['Method'] == m]['Novelty'].iloc[0]) for m in methods]
    perspective_vals = [float(llm_df[llm_df['Method'] == m]['Perspective'].iloc[0]) for m in methods]

    bars1 = ax2.bar(x - width/2, novelty_vals, width, label='Novelty\n(1-5 scale)',
                    color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, perspective_vals, width, label='Perspective\n(1-5 scale)',
                    color='#FFD93D', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.05,
                f'{novelty_vals[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.05,
                f'{perspective_vals[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Recommendation Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score (1-5 scale)', fontsize=11, fontweight='bold')
    ax2.set_title('2. LLM Evaluation Metrics', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 5.5)

    # Chart 3: Diversity Method Metrics (Echo Chamber % & Framing Diff %)
    ax3 = fig.add_subplot(gs[1, 2])
    echo_pcts = []
    for m in methods:
        val = centroid_df[centroid_df['Method'] == m]['Echo_Chamber_%'].iloc[0]
        echo_pcts.append(float(str(val).rstrip('%')) if isinstance(val, str) else float(val))

    framing_vals = []
    for m in methods:
        val = llm_df[llm_df['Method'] == m]['Framing_Diff_%'].iloc[0]
        framing_vals.append(float(str(val).rstrip('%')) if isinstance(val, str) else float(val))

    bars1 = ax3.bar(x - width/2, echo_pcts, width, label='Echo Chamber %\n(lower is better)',
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x + width/2, framing_vals, width, label='Framing Diff %\n(higher is better)',
                    color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                f'{echo_pcts[i]:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                f'{framing_vals[i]:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Recommendation Method', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('3. Diversity & Framing Metrics', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)

    # Print summary statistics instead of showing in plot
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*80)

    # Calculate rankings for summary
    metric_names = ['Echo Score', 'Rec Diversity', 'LLM Novelty', 'LLM Perspective',
                    'Echo Chamber %', 'Framing Diff %']

    # Echo Score rankings (lower is better)
    echo_rankings = [sorted(methods, key=lambda m: echo_scores[methods.index(m)]).index(m) + 1 for m in methods]
    # Rec Diversity rankings (higher is better)
    rec_div_rankings = [sorted(methods, key=lambda m: rec_divs[methods.index(m)], reverse=True).index(m) + 1 for m in methods]
    # Novelty rankings (higher is better)
    novelty_rankings = [sorted(methods, key=lambda m: novelty_vals[methods.index(m)], reverse=True).index(m) + 1 for m in methods]
    # Perspective rankings (higher is better)
    perspective_rankings = [sorted(methods, key=lambda m: perspective_vals[methods.index(m)], reverse=True).index(m) + 1 for m in methods]
    # Echo Chamber % rankings (lower is better)
    echo_pct_rankings = [sorted(methods, key=lambda m: echo_pcts[methods.index(m)]).index(m) + 1 for m in methods]
    # Framing % rankings (higher is better)
    framing_rankings = [sorted(methods, key=lambda m: framing_vals[methods.index(m)], reverse=True).index(m) + 1 for m in methods]

    rankings_matrix = np.array([
        echo_rankings,
        rec_div_rankings,
        novelty_rankings,
        perspective_rankings,
        echo_pct_rankings,
        framing_rankings
    ])

    # Count wins and losses
    wins = {method: 0 for method in methods}
    losses = {method: 0 for method in methods}
    for i in range(len(metric_names)):
        for j in range(len(methods)):
            if rankings_matrix[i, j] == 1:
                wins[methods[j]] += 1
            elif rankings_matrix[i, j] == 3:
                losses[methods[j]] += 1

    # Print summary for each method
    for method in methods:
        win_count = wins[method]
        loss_count = losses[method]
        if win_count > loss_count:
            perf = "✓ Better Overall"
        elif loss_count > win_count:
            perf = "✗ Worse Overall"
        else:
            perf = "~ Mixed Performance"
        print(f"\n{method}:")
        print(f"  Wins (rank 1): {win_count} metrics")
        print(f"  Losses (rank 3): {loss_count} metrics")
        print(f"  Overall: {perf}")

    print("\n" + "="*80)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive comparison: {save_path}")
    plt.close()

    # Also save the table as CSV
    table_csv_path = save_path.replace('.png', '_table.csv')
    table_df.to_csv(table_csv_path, index=False)
    print(f"Saved comprehensive table: {table_csv_path}")

def create_overall_ranking_visualization(centroid_df, llm_df, diversity_df, save_path="eval_results/overall_ranking.png"):
    """Create a standalone visualization showing overall performance rankings"""
    methods = centroid_df['Method'].tolist()

    # Get all metric values
    echo_scores = [float(centroid_df[centroid_df['Method'] == m]['Echo_Score'].iloc[0]) for m in methods]
    rec_divs = [float(centroid_df[centroid_df['Method'] == m]['Rec_Div'].iloc[0]) for m in methods]
    novelty_vals = [float(llm_df[llm_df['Method'] == m]['Novelty'].iloc[0]) for m in methods]
    perspective_vals = [float(llm_df[llm_df['Method'] == m]['Perspective'].iloc[0]) for m in methods]

    echo_pcts = []
    for m in methods:
        val = centroid_df[centroid_df['Method'] == m]['Echo_Chamber_%'].iloc[0]
        echo_pcts.append(float(str(val).rstrip('%')) if isinstance(val, str) else float(val))

    framing_vals = []
    for m in methods:
        val = llm_df[llm_df['Method'] == m]['Framing_Diff_%'].iloc[0]
        framing_vals.append(float(str(val).rstrip('%')) if isinstance(val, str) else float(val))

    # Calculate rankings
    metric_names = ['Echo Score\n(lower=better)', 'Rec Diversity\n(higher=better)',
                    'LLM Novelty\n(higher=better)', 'LLM Perspective\n(higher=better)',
                    'Echo Chamber %\n(lower=better)', 'Framing Diff %\n(higher=better)']

    echo_rankings = [sorted(methods, key=lambda m: echo_scores[methods.index(m)]).index(m) + 1 for m in methods]
    rec_div_rankings = [sorted(methods, key=lambda m: rec_divs[methods.index(m)], reverse=True).index(m) + 1 for m in methods]
    novelty_rankings = [sorted(methods, key=lambda m: novelty_vals[methods.index(m)], reverse=True).index(m) + 1 for m in methods]
    perspective_rankings = [sorted(methods, key=lambda m: perspective_vals[methods.index(m)], reverse=True).index(m) + 1 for m in methods]
    echo_pct_rankings = [sorted(methods, key=lambda m: echo_pcts[methods.index(m)]).index(m) + 1 for m in methods]
    framing_rankings = [sorted(methods, key=lambda m: framing_vals[methods.index(m)], reverse=True).index(m) + 1 for m in methods]

    rankings_matrix = np.array([
        echo_rankings,
        rec_div_rankings,
        novelty_rankings,
        perspective_rankings,
        echo_pct_rankings,
        framing_rankings
    ])

    # Create standalone figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a cleaner color scheme - blue tones
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#4CAF50', '#FFC107', '#F44336']  # Green, Amber, Red
    n_bins = 3
    cmap = LinearSegmentedColormap.from_list('ranking', colors_list, N=n_bins)

    im = ax.imshow(rankings_matrix, cmap=cmap, aspect='auto', vmin=0.5, vmax=3.5)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.set_yticklabels(metric_names, fontsize=11)

    # Add text annotations with rankings
    for i in range(len(metric_names)):
        for j in range(len(methods)):
            rank = rankings_matrix[i, j]
            # Use white text for better contrast
            ax.text(j, i, f'{int(rank)}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color='white')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3], shrink=0.8)
    cbar.set_label('Ranking', fontsize=12, fontweight='bold')
    cbar.set_ticklabels(['1 (Best)', '2 (Middle)', '3 (Worst)'])

    ax.set_title('Overall Performance Rankings Across All Evaluation Methods',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Recommendation Method', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('Evaluation Metric', fontsize=13, fontweight='bold', labelpad=15)

    # Print summary to console instead of showing on plot
    wins = {method: 0 for method in methods}
    losses = {method: 0 for method in methods}
    for i in range(len(metric_names)):
        for j in range(len(methods)):
            if rankings_matrix[i, j] == 1:
                wins[methods[j]] += 1
            elif rankings_matrix[i, j] == 3:
                losses[methods[j]] += 1

    print("\n" + "="*80)
    print("OVERALL RANKING SUMMARY")
    print("="*80)
    for method in methods:
        win_count = wins[method]
        loss_count = losses[method]
        if win_count > loss_count:
            perf = "✓ Better Overall"
        elif loss_count > win_count:
            perf = "✗ Worse Overall"
        else:
            perf = "~ Mixed Performance"
        print(f"{method}: {win_count} wins, {loss_count} losses ({perf})")
    print("="*80)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved overall ranking visualization: {save_path}")
    plt.close()

def plot_combined_comparison(centroid_df, llm_df, diversity_df, save_path="eval_results/combined_comparison.png"):
    """Create a combined comparison across all 3 evaluation methods"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = centroid_df['Method'].values

    # Normalize scores to 0-1 for comparison
    # 1. Centroid method: Use Echo_Score (inverted - lower is better, so invert for "diversity" metric)
    centroid_df['Echo_Score'] = centroid_df['Echo_Score'].astype(float)
    echo_norm = 1 - ((centroid_df['Echo_Score'] - centroid_df['Echo_Score'].min()) /
                     (centroid_df['Echo_Score'].max() - centroid_df['Echo_Score'].min() + 1e-10))

    # 2. LLM Novelty: normalize 1-5 scale to 0-1
    llm_df['Novelty'] = llm_df['Novelty'].astype(float)
    novelty_norm = (llm_df['Novelty'] - 1) / 4  # Scale from 1-5 to 0-1

    # 3. Diversity method: Use Rec_Div
    diversity_df['Rec_Div'] = diversity_df['Rec_Div'].astype(float)
    diversity_norm = (diversity_df['Rec_Div'] - diversity_df['Rec_Div'].min()) / \
                     (diversity_df['Rec_Div'].max() - diversity_df['Rec_Div'].min() + 1e-10)

    x = np.arange(len(methods))
    width = 0.25

    # Combined bar chart - only 3 metrics
    axes[0, 0].bar(x - width, echo_norm, width, label='Centroid (Echo Score)', color='#FF6B6B', alpha=0.8)
    axes[0, 0].bar(x, novelty_norm, width, label='LLM Novelty', color='#4ECDC4', alpha=0.8)
    axes[0, 0].bar(x + width, diversity_norm, width, label='Diversity (Rec_Div)', color='#95E1D3', alpha=0.8)
    axes[0, 0].set_xlabel('Method', fontsize=12)
    axes[0, 0].set_ylabel('Normalized Score (0-1)', fontsize=12)
    axes[0, 0].set_title('Combined Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=15)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Radar/spider chart - only 3 metrics
    metrics = ['Centroid\n(Echo)', 'LLM\nNovelty', 'Diversity\n(Rec_Div)']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for idx, method in enumerate(methods):
        values = [
            echo_norm.iloc[idx],
            novelty_norm.iloc[idx],
            diversity_norm.iloc[idx]
        ]
        values += values[:1]  # Complete the circle

        axes[0, 1].plot(angles, values, 'o-', linewidth=2, label=method)
        axes[0, 1].fill(angles, values, alpha=0.25)

    axes[0, 1].set_xticks(angles[:-1])
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('Metrics Profile Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Summary table
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    summary_data = {
        'Method': methods,
        'Echo Score': [f"{v:.3f}" for v in centroid_df['Echo_Score']],
        'LLM Novelty': [f"{v:.2f}" for v in llm_df['Novelty']],
        'Rec Diversity': [f"{v:.3f}" for v in diversity_df['Rec_Div']]
    }
    summary_df = pd.DataFrame(summary_data)
    table = axes[1, 0].table(cellText=summary_df.values, colLabels=summary_df.columns,
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 0].set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)

    # Ranking visualization - only 3 metrics
    rankings = {
        'Centroid': echo_norm.rank(ascending=False),
        'LLM Novelty': novelty_norm.rank(ascending=False),
        'Diversity': diversity_norm.rank(ascending=False)
    }
    ranking_df = pd.DataFrame(rankings, index=methods)

    # Fill NaN values with 2 (middle rank) if any exist
    ranking_df = ranking_df.fillna(2.0)

    # Create heatmap
    im = axes[1, 1].imshow(ranking_df.values, cmap='RdYlGn', aspect='auto', vmin=1, vmax=3)
    axes[1, 1].set_xticks(range(len(ranking_df.columns)))
    axes[1, 1].set_xticklabels(ranking_df.columns, rotation=45, ha='right')
    axes[1, 1].set_yticks(range(len(ranking_df.index)))
    axes[1, 1].set_yticklabels(ranking_df.index)
    axes[1, 1].set_title('Ranking Heatmap (1=Best, 3=Worst)', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(ranking_df.index)):
        for j in range(len(ranking_df.columns)):
            val = ranking_df.iloc[i, j]
            if pd.notna(val):
                axes[1, 1].text(j, i, f'{int(val)}',
                               ha='center', va='center', fontweight='bold', color='white', fontsize=12)

    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    print("Loading evaluation results...")
    centroid_df, llm_df, diversity_df = load_results()

    print("\nGenerating visualizations...")

    # Create output directories for each method
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "eval_results")

    # Create method-specific folders
    centroid_dir = os.path.join(output_dir, "centroid_method")
    llm_dir = os.path.join(output_dir, "llm_method")
    diversity_dir = os.path.join(output_dir, "diversity_method")

    for dir_path in [output_dir, centroid_dir, llm_dir, diversity_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Generate individual visualizations in their respective folders
    print("\nGenerating centroid method visualizations...")
    plot_centroid_results(centroid_df, os.path.join(centroid_dir, "centroid_visualizations.png"))

    print("\nGenerating LLM method visualizations...")
    plot_llm_results(llm_df, os.path.join(llm_dir, "llm_visualizations.png"))

    print("\nGenerating diversity method visualizations...")
    plot_diversity_results(diversity_df, os.path.join(diversity_dir, "diversity_visualizations.png"))

    # Also keep the old diversity visualization in the main folder (as requested)
    plot_diversity_results(diversity_df, os.path.join(output_dir, "diversity_visualizations.png"))

    # Create additional individual plots for each method
    print("\nGenerating individual method plots...")

    # Individual centroid plots
    plot_centroid_individual(centroid_df, centroid_dir)

    # Individual LLM plots
    plot_llm_individual(llm_df, llm_dir)

    # Individual diversity plots
    plot_diversity_individual(diversity_df, diversity_dir)

    # Generate combined comparison in main folder
    print("\nGenerating combined comparison...")
    create_combined_table_and_visualization(centroid_df, llm_df, diversity_df,
                                            os.path.join(output_dir, "combined_comparison.png"))

    # Generate standalone overall ranking visualization
    print("\nGenerating overall ranking visualization...")
    create_overall_ranking_visualization(centroid_df, llm_df, diversity_df,
                                        os.path.join(output_dir, "overall_ranking.png"))

    # Also keep the old combined comparison for backward compatibility
    plot_combined_comparison(centroid_df, llm_df, diversity_df,
                            os.path.join(output_dir, "combined_comparison_old.png"))

    print("\nAll visualizations saved:")
    print(f"  - Centroid method: {centroid_dir}/")
    print(f"  - LLM method: {llm_dir}/")
    print(f"  - Diversity method: {diversity_dir}/")
    print(f"  - Combined: {output_dir}/combined_comparison.png")

if __name__ == "__main__":
    main()
