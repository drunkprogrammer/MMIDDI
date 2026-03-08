import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import os
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix


def draw_top_confused_pairs(y_true, y_pred, event_num, saved_path, task, top_k=20):
    """
    Identifies and plots the top K most frequent errors (e.g., True Class A -> Pred Class B).
    """

    # 1. Calculate Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(event_num))

    # 2. Extract Off-Diagonal Elements (Errors only)
    pairs = []
    for i in range(event_num):
        for j in range(event_num):
            if i != j and cm[i, j] > 0:  # Ignore diagonal and zeros
                pairs.append({
                    'True Class': i,
                    'Predicted Class': j,
                    'Count': cm[i, j],
                    'Label': f'True {i} -> Pred {j}'
                })

    # 3. Create DataFrame and Sort
    df_errors = pd.DataFrame(pairs)
    if df_errors.empty:
        print("No errors found!")
        return

    df_errors = df_errors.sort_values(by='Count', ascending=False).head(top_k)

    # 4. Plot
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    # Bar Chart
    ax = sns.barplot(data=df_errors, x='Count', y='Label', palette='magma')

    # Annotate bars with values
    for i in ax.containers:
        ax.bar_label(i, padding=3, fontsize=12)

    plt.title(f'Top {top_k} Most Confusing Pairs ({task})', fontsize=18)
    plt.xlabel('Number of Misclassified Samples', fontsize=14)
    plt.ylabel('Confusion Pair', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save
    filename = f'{task}_top{top_k}_errors.png'
    plt.tight_layout()
    plt.savefig(os.path.join(saved_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Top-{top_k} error plot for {task}")

def draw_confusion_network(y_true, y_pred, event_num, saved_path, task, min_error_threshold=0):
    """
    - Shows ALL data (min_error_threshold=0 by default).
    - Uses non-linear scaling for width and opacity to keep the graph clean.
    - Sorts edges so heavy errors pop out on top.
    """

    print(f"Generating Nature-style Network Graph for {task}...")

    # 1. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(event_num))

    # 2. Build Graph
    G = nx.DiGraph()
    for i in range(event_num):
        G.add_node(i)

    # 3. Collect ALL Edges (No Top-K filtering, showing all data)
    all_edges = []
    max_val = 0

    for i in range(event_num):
        for j in range(event_num):
            if i != j:
                count = cm[i, j]
                if count > min_error_threshold:
                    all_edges.append((i, j, count))
                    if count > max_val: max_val = count

    if not all_edges:
        print("No errors found.")
        return

    # 4. SORTING IS CRITICAL
    # We sort ascending. Why? Because we draw lines in order.
    # We want faint lines drawn FIRST (background), and strong lines LAST (foreground).
    all_edges.sort(key=lambda x: x[2], reverse=False)

    for u, v, w in all_edges:
        G.add_edge(u, v, weight=w)

    # 5. Remove perfectly isolated nodes (optional, keeps graph compact)
    # If a class was never misclassified and never confused, we can hide it to save space.
    nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)

    # 6. Layout
    pos = nx.circular_layout(G)

    # Large canvas for high resolution
    plt.figure(figsize=(20, 20), facecolor='white')

    # --- STYLE CONFIGURATION (Nature Journal Style) ---
    # Colormap: Spectral_r goes from Blue (Low) -> Yellow -> Red (High)
    # It provides excellent contrast on white paper.
    cmap = plt.cm.Spectral_r

    # Extract weights
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Normalization
    norm = mcolors.Normalize(vmin=0, vmax=max_val)

    # --- NON-LINEAR SCALING (The Secret to Clarity) ---
    # Instead of linear width, we raise normalized weight to a power (e.g., 2.5).
    # This suppresses the visual noise of small errors while highlighting big ones.

    edge_colors = [cmap(norm(w)) for w in weights]

    # Width: Range from 0.5px (ghost) to 6.0px (heavy)
    widths = [0.5 + 5.5 * ((w / max_val) ** 2) for w in weights]

    # Opacity: Small errors are very transparent (0.2), Big errors are opaque (1.0)
    alphas = [0.2 + 0.8 * ((w / max_val) ** 2) for w in weights]

    # 7. Draw Nodes (Clean, Scientific Look)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=900,
        node_color='white',  # Clean white background
        edgecolors='#333333',  # Dark grey border
        linewidths=1.5
    )

    # Node Labels (Helvetica/Arial style)
    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_family='sans-serif',
        font_weight='bold',
        font_color='black'
    )

    # 8. Draw Edges Iteratively to handle alpha (NetworkX limitation)
    ax = plt.gca()
    for i, (u, v) in enumerate(G.edges()):
        weight = weights[i]

        # Calculate visual properties for this specific edge
        color = edge_colors[i]
        width = widths[i]
        alpha = alphas[i]

        # Draw curved arrow
        # rad=0.2 gives a nice arc. Increase to 0.3 if still too crowded.
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color=[color],
            arrowstyle='-|>',
            arrowsize=15 + (15 * (weight / max_val)),  # Scale arrow head size too
            connectionstyle="arc3,rad=0.2"
        )

    # 9. Add Colorbar (Scientific Style)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label('Count of Misclassified Samples', fontsize=16, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(0.5)  # Thin border

    # 10. Title and Save
    plt.title(f'Confusion Network ({task})', fontsize=24, pad=20, fontname='sans-serif')
    plt.axis('off')

    figurename = f'{task}_confusion_network_nature.png'
    filename = os.path.join(saved_path, figurename)
    plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved optimized network graph: {filename}")


def draw_top10_confusion_network(y_true, y_pred, event_num, saved_path, task, min_error_threshold=0,
                                 top_n_highlight=10):
    """
    Final Refined Network Graph.
    - Uses Geometry-aware curvature to separate overlapping lines.
    - Draws faint context lines first, then bold top 10 lines.
    - Ensures all 10 top errors are distinct.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import matplotlib.colors as mcolors
    from sklearn.metrics import confusion_matrix

    print(f"Generating Anti-Overlap Network Graph for {task}...")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(event_num))

    G = nx.DiGraph()
    for i in range(event_num):
        G.add_node(i)

    # 1. Collect ALL Edges
    all_edges = []
    for i in range(event_num):
        for j in range(event_num):
            if i != j:
                count = cm[i, j]
                if count > min_error_threshold:
                    all_edges.append((i, j, count))

    if not all_edges:
        print("No errors found.")
        return

    # 2. Sort Edges by Weight (Highest Error First)
    all_edges.sort(key=lambda x: x[2], reverse=True)

    # 3. Split into "Top N" (Focus) and "Others" (Context)
    top_edges = all_edges[:top_n_highlight]
    other_edges = all_edges[top_n_highlight:]

    # Add edges to graph structure
    for u, v, w in all_edges:
        G.add_edge(u, v, weight=w)

    nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)

    pos = nx.circular_layout(G)

    plt.figure(figsize=(20, 20), facecolor='white')

    # ==========================================
    # LAYER 1: The Context (Faint Grey)
    # ==========================================
    if other_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, w in other_edges],
            width=0.5,
            alpha=0.1,
            edge_color='#CCCCCC',
            arrowstyle='-|>',
            arrowsize=5,
            connectionstyle="arc3,rad=0.1"  # Flat curves for background
        )

    # ==========================================
    # LAYER 2: The Focus (Top 10) - Geometric Separation
    # ==========================================
    top_weights = [w for u, v, w in top_edges]
    max_val = max(top_weights) if top_weights else 1
    min_val = min(top_weights) if top_weights else 0

    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=min_val * 0.5, vmax=max_val)

    ax = plt.gca()

    # We iterate through the Top Edges.
    # We maintain a counter for edges connecting to specific nodes to shift them.
    for i, (u, v, w) in enumerate(reversed(top_edges)):  # Draw smallest of top 10 first

        color = cmap(norm(w))

        # Scale Width and Arrow
        width = 1.5 + 4.0 * (w / max_val)
        arrow_size = 15 + 15 * (w / max_val)

        # --- ANTI-OVERLAP LOGIC ---
        # Calculate 'distance' in node indices (roughly)
        dist = abs(u - v)
        if dist > event_num / 2: dist = event_num - dist

        # Base curvature: Larger distance = flatter curve, Shorter distance = rounder curve
        # This keeps short links from disappearing inside the nodes
        base_rad = 0.4 if dist < 3 else 0.2

        # Add a "lane shift" based on the index i to force separation
        # We alternate adding/subtracting curvature to fan them out
        shift = (i * 0.05)
        final_rad = base_rad + shift

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            alpha=0.9,
            edge_color=[color],
            arrowstyle='-|>',
            arrowsize=arrow_size,
            connectionstyle=f"arc3,rad={final_rad}"  # Unique curve for every line
        )

    # ==========================================
    # LAYER 3: Nodes
    # ==========================================
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color='white',
        edgecolors='black',
        linewidths=2.0
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=13,
        font_family='sans-serif',
        font_weight='bold',
        font_color='black'
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label('Count of Misclassified Samples', fontsize=16, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    plt.title(f'Confusion Network ({task}) - Top {top_n_highlight} Errors', fontsize=24, pad=20)
    plt.axis('off')

    filename = f'{task}_confusion_network_top{top_n_highlight}_final.png'
    save_full = os.path.join(saved_path, filename)
    plt.savefig(save_full, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved Final Optimized Network to: {save_full}")


def draw_focused_top_nodes_network(y_true, y_pred, event_num, saved_path, task, top_n_highlight=10):
    """
    Draws a focused network graph.
    FIX: Draws edges ON TOP of nodes to ensure arrows are never masked.
    FIX: Darker minimum color for visibility.
    """

    print(f"Generating Focused Top-{top_n_highlight} Node Graph for {task}...")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(event_num))

    # 1. Collect and Sort Edges
    all_edges = []
    for i in range(event_num):
        for j in range(event_num):
            if i != j and cm[i, j] > 0:
                all_edges.append((i, j, cm[i, j]))

    all_edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = all_edges[:top_n_highlight]

    if not top_edges:
        print("No errors found.")
        return

    # 2. Build Graph
    G = nx.DiGraph()
    relevant_nodes = set()
    for u, v, w in top_edges:
        relevant_nodes.add(u)
        relevant_nodes.add(v)

    for node in relevant_nodes:
        G.add_node(node)

    for u, v, w in top_edges:
        G.add_edge(u, v, weight=w)

    pos = nx.circular_layout(G)

    plt.figure(figsize=(12, 12), facecolor='white')

    # --- Setup Colors ---
    weights = [w for u, v, w in top_edges]
    max_val = max(weights)
    min_val = min(weights)

    # IMPROVEMENT 1: Color Visibility
    # Use a custom normalization that starts "higher" in the Red spectrum.
    # This makes the lowest error (min_val) appear Dark Orange instead of Pale Pink.
    cmap = plt.cm.Reds
    # We map our data range [min, max] to the colormap range [0.3, 1.0]
    # This skips the white/faint part of the Reds colormap.
    norm = mcolors.Normalize(vmin=0, vmax=max_val)  # Standard norm

    # Custom color getter to ensure minimum darkness
    def get_color(weight):
        # Normalize 0..1 based on data
        n_val = (weight - min_val) / (max_val - min_val + 1e-9)
        # Shift to 0.4..1.0 range (Darker start)
        shifted_val = 0.4 + (0.6 * n_val)
        return cmap(shifted_val)

    ax = plt.gca()

    # --- STEP 1: Draw Nodes FIRST (Background) ---
    MY_NODE_SIZE = 2000
    nx.draw_networkx_nodes(
        G, pos,
        node_size=MY_NODE_SIZE,
        node_color='white',
        edgecolors='#333333',
        linewidths=2.5
    )

    # --- STEP 2: Draw Edges SECOND (Foreground) ---
    # This prevents the node from covering the arrowhead.

    for i, (u, v, w) in enumerate(reversed(top_edges)):
        color = get_color(w)

        # Width: 2.0 to 6.0
        width = 2.0 + 4.0 * (w / max_val)

        # IMPROVEMENT 2: Larger Arrows
        # Base size 15 (was 20), scaling up to 45
        arrow_size = 15 + 20 * (w / max_val)

        # Curvature Logic
        # Spread arcs slightly to avoid overlap
        rad = 0.3 + (i % 5) * 0.05

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            alpha=1.0,  # Full opacity
            edge_color=[color],
            arrowstyle='-|>',
            arrowsize=arrow_size,
            node_size=MY_NODE_SIZE,  # Helps calculated cutoff
            connectionstyle=f"arc3,rad={rad}"
        )

    # --- STEP 3: Draw Labels LAST ---
    nx.draw_networkx_labels(
        G, pos,
        font_size=16,
        font_family='sans-serif',
        font_weight='bold',
        font_color='black'
    )

    # --- Colorbar ---
    # Create a dummy mappable for the colorbar that matches our custom range
    # We use a custom cmap subset to match the get_color logic visually
    from matplotlib.colors import LinearSegmentedColormap
    colors_for_bar = cmap(np.linspace(0.4, 1.0, 256))  # Match the 0.4 start
    new_cmap = LinearSegmentedColormap.from_list('DarkerReds', colors_for_bar)

    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=mcolors.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label('Count of Misclassified Samples', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    plt.title(f'Focused Confusion Network ({task})\nTop {top_n_highlight} Errors', fontsize=20, pad=20)
    plt.axis('off')

    filename = f'{task}_confusion_network_focused_top{top_n_highlight}.png'
    save_full = os.path.join(saved_path, filename)
    plt.savefig(save_full, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved Fixed Network (Visible Arrows) to: {save_full}")


def draw_sub_class_accuracy_curve(y_true, y_pred, event_num, task, saved_path):
    """
    Calculates global accuracy, but draws the accuracy curves in split sub-figures
    (parts) similar to the confusion matrix logic.
    """
    classes_all = np.arange(event_num)
    cm = confusion_matrix(y_true, y_pred, labels=classes_all)
    class_totals = cm.sum(axis=1)
    class_accuracies = cm.diagonal() / (class_totals + 1e-10) * 100

    # Calculate Global Mean Accuracy (only for classes that actually had samples)
    valid_classes_mask = class_totals >= 0
    if np.any(valid_classes_mask):
        global_mean_acc = np.mean(class_accuracies[valid_classes_mask])
    else:
        global_mean_acc = 0.0

    num_figures = 1
    if event_num == 65:
        num_figures = 5  # 13 classes per figure
    elif event_num == 64:
        num_figures = 8  # 8 classes per figure
    elif event_num == 100:
        num_figures = 10  # 10 classes per figure

    step_size = event_num // num_figures

    for k in range(num_figures):
        start_idx = k * step_size
        end_idx = start_idx + step_size

        sub_classes = classes_all[start_idx:end_idx]
        sub_accuracies = class_accuracies[start_idx:end_idx]

        plt.figure(figsize=(12, 6))
        plt.plot(sub_classes, sub_accuracies, marker='o', linestyle='-', color='b',
                 linewidth=2, markersize=8, label='Class Accuracy')
        plt.axhline(y=global_mean_acc, color='r', linestyle='--', linewidth=2,
                    label=f'Global Mean ({global_mean_acc:.2f}%)')

        plt.title(f'Accuracy Profile (Part {k + 1}/{num_figures}) - Classes {start_idx}-{end_idx - 1}', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Class Index', fontsize=12)

        plt.xticks(sub_classes, fontsize=10)
        plt.ylim(0, 105)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        filename = f'{task}_part{k + 1}_accuracy_curve.png'
        save_file = os.path.join(saved_path, filename)

        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    print(f"Saved {num_figures} sub-accuracy curves for task: {task}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    file_path = 'task1_evaluation_data.npz'
    save_dir = './visualizations/'

    # Parameters
    TOP_K = 20  # How many pairs to show in the bar chart
    NET_THRESHOLD = 5  # Minimum errors to show an arrow in the network graph

    # --- LOAD DATA ---
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
    else:
        print(f"Loading data from {file_path}...")
        data = np.load(file_path)

        y_true = data['y_true']
        y_pred = data['y_pred']

        # Extract scalar from array
        if data['event_num'].size == 1:
            event_num = int(data['event_num'][0])
        else:
            event_num = int(data['event_num'])

        # Determine Task Name from filename (optional)
        task_name = "task1"
        if "task2" in file_path: task_name = "task2"

        # --- GENERATE PLOTS ---
        draw_top_confused_pairs(y_true, y_pred, event_num, save_dir, task_name, top_k=TOP_K)
        draw_confusion_network(y_true, y_pred, event_num, save_dir, task_name, min_error_threshold=NET_THRESHOLD)
        draw_top10_confusion_network(y_true, y_pred, event_num, save_dir, task_name, min_error_threshold=0, top_n_highlight=10)
        draw_focused_top_nodes_network(y_true, y_pred, event_num, save_dir, task_name, top_n_highlight=10)
        # draw_sub_class_accuracy_curve(y_true, y_pred, event_num, task_name, save_dir)

        print("Done!")