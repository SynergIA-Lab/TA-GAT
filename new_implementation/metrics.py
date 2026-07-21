import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import linregress
from scipy.stats import wilcoxon as _wilcoxon
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    roc_curve, auc, average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt


def _approx_diameter(G_sub: nx.Graph) -> int:
    """
    Estimates the graph diameter using the double-sweep heuristic (2 BFS calls).

    FIX (O(N·(N+E)) → O(N+E)): nx.diameter performs BFS from every node to find the
    true diameter, which with N=1500 nodes and dense edges can take minutes per call.
    The double-sweep runs only 2 BFS traversals and returns a tight lower-bound estimate
    (exact on trees and near-exact on most real networks).

    1. BFS from an arbitrary source to find the farthest node (far1).
    2. BFS from far1 to find the farthest node from there.
    3. Return the eccentricity of far1 as the diameter estimate.

    Args:
        G_sub (nx.Graph): A connected subgraph (e.g., the largest connected component).

    Returns:
        int: Estimated diameter (lower bound of the true diameter).
    """
    if len(G_sub) == 0:
        return 0
    source = next(iter(G_sub.nodes()))
    lengths1 = nx.single_source_shortest_path_length(G_sub, source)
    far1 = max(lengths1, key=lengths1.get)
    lengths2 = nx.single_source_shortest_path_length(G_sub, far1)
    return int(max(lengths2.values()))

##################################################
# PHASE 1: NETWORK TOPOLOGICAL PROPERTIES       #
##################################################

def compute_scale_free_fit(G: nx.Graph) -> float:
    """
    Computes the R-squared (R2) fit of the network's degree distribution to a power law.

    Fits the log10-transformed degree frequency histogram against the log10-transformed 
    degrees using simple linear regression. A higher R2 score indicates a higher 
    scale-free topology alignment.

    Args:
        G (nx.Graph): The network graph to analyze.

    Returns:
        float: The R-squared coefficient of determination. Returns 0.0 if the graph 
            has fewer than two unique degree values.
    """
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return 0.0
    
    degree_counts = nx.degree_histogram(G)
    x = []
    y = []
    
    total_nodes = len(degrees)
    
    for degree, count in enumerate(degree_counts):
        if count > 0 and degree > 0:
            x.append(np.log10(degree))
            y.append(np.log10(count / total_nodes))
            
    if len(x) < 2:
        return 0.0
        
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return float(r_value ** 2)

def compute_network_topology_metrics(G: nx.Graph) -> dict[str, any]:
    """
    Computes a comprehensive set of topological metrics for a given network.

    Calculates fundamental network metrics including node and edge count, 
    average node degree, average clustering coefficient, network density, 
    scale-free power-law fit (R2), and the diameter of the largest connected component (LCC).

    Args:
        G (nx.Graph): The network graph to analyze.

    Returns:
        dict[str, any]: A dictionary containing the following calculated metrics:
            - 'Nodes' (int): Total number of nodes.
            - 'Edges' (int): Total number of edges.
            - 'Scale-Free R²' (float): Power-law fit R2 score.
            - 'Avg Degree' (float): Mean degree of the nodes.
            - 'Clustering Coef' (float): Average local clustering coefficient.
            - 'Diameter (LCC)' (int): Diameter of the largest connected component.
            - 'Density' (float): Global edge density.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    r2 = compute_scale_free_fit(G)
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
    avg_clustering = nx.average_clustering(G)
    
    if num_nodes > 0 and num_edges > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        diameter = _approx_diameter(G_sub)  # FIX: double-sweep BFS (2 BFS vs N BFS)
    else:
        diameter = 0
        
    density = nx.density(G)

    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Scale-Free R²": r2,
        "Avg Degree": avg_degree,
        "Clustering Coef": avg_clustering,
        "Diameter (LCC)": diameter,
        "Density": density
    }

##################################################
# PHASE 2: GOLD STANDARD VALIDATION & BENCHMARKS#
##################################################

def evaluate_against_gold_standard(pred_adj: np.ndarray, truth_adj: np.ndarray, gene_names: list[str], evaluable_genes_set: set[str]) -> dict[str, any]:
    """
    Evaluates a predicted network against an experimental gold standard.

    Implements the 'Fair Evaluation' methodology, restricting all confusion matrix 
    calculations (TP, FP, TN, FN) to the sub-network of genes present in both 
    the local differentially expressed gene list and the gold standard reference set.
    Computes Accuracy, Precision, Recall, and F1-Score.

    Args:
        pred_adj (np.ndarray): Binary predicted adjacency matrix of shape (num_genes, num_genes).
        truth_adj (np.ndarray): Binary gold standard adjacency matrix of shape (num_genes, num_genes).
        gene_names (list[str]): List of gene names corresponding to the indices of the adjacency matrices.
        evaluable_genes_set (set[str]): Set of gene names present in both the DEGs and the gold standard.

    Returns:
        dict[str, any]: A dictionary containing:
            - 'Accuracy' (float): Fraction of correctly predicted edges and non-edges.
            - 'Precision' (float): Fraction of predicted edges that are true.
            - 'Recall' (float): Fraction of true edges that are predicted.
            - 'F1-Score' (float): Harmonic mean of Precision and Recall.
            - 'Pred_Edges' (int): Count of predicted edges in the sub-network.
            - 'True_Edges' (int): Count of true edges in the sub-network.
            - 'Common_Genes' (int): Count of evaluable genes.
    """
    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    
    if len(eval_indices) < 2:
        return {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "Pred_Edges": 0, "True_Edges": 0, "Common_Genes": 0}
    
    sub_pred = pred_adj[np.ix_(eval_indices, eval_indices)]
    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]
    
    upper_mask = np.triu(np.ones_like(sub_truth, dtype=bool), k=1)
    
    y_pred = sub_pred[upper_mask].astype(int)
    y_true = sub_truth[upper_mask].astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "Accuracy":     acc,
        "Precision":    prec,
        "Recall":       rec,
        "F1-Score":     f1,
        "AUC":          0.0,   # placeholder; filled by compute_auc_against_gs in run_condition_pipeline
        "Pred_Edges":   int(np.sum(y_pred)),
        "True_Edges":   int(np.sum(y_true)),
        "Common_Genes": len(eval_indices)
    }


def compute_auc_against_gs(
    score_adj: np.ndarray,
    truth_adj: np.ndarray,
    gene_names: list[str],
    evaluable_genes_set: set[str]
) -> float:
    """
    Computes the AUC-ROC of a continuous edge-score matrix against a binary gold standard.

    FIX: Previously, AUC was only visualised in ROC plots; it was not included in the
    summary comparison table. This function enables AUC to be stored in the results dict
    and surfaced in print_final_summary_tables alongside Precision, Recall, and F1.

    Restricts the evaluation to the 'fair evaluation' sub-network: only gene pairs where
    both genes are present in the evaluable set (intersection of DEGs and gold-standard genes).

    Args:
        score_adj (np.ndarray): Continuous predicted edge-score matrix of shape (N, N).
        truth_adj (np.ndarray): Binary gold-standard adjacency matrix of shape (N, N).
        gene_names (list[str]): Gene names corresponding to matrix indices.
        evaluable_genes_set (set[str]): Genes present in both DEG list and gold standard.

    Returns:
        float: Area under the ROC curve. Returns 0.0 if the sub-network has fewer than
            2 evaluable genes or no positive edges in the gold standard.
    """
    from sklearn.metrics import roc_auc_score

    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    if len(eval_indices) < 2:
        return 0.0

    sub_score = score_adj[np.ix_(eval_indices, eval_indices)]
    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]

    upper_mask = np.triu(np.ones_like(sub_truth, dtype=bool), k=1)
    y_score = np.nan_to_num(sub_score[upper_mask], nan=0.0, posinf=0.0, neginf=0.0)
    y_true  = sub_truth[upper_mask].astype(int)

    if y_true.sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def compute_auprc_against_gs(
    score_adj: np.ndarray,
    truth_adj: np.ndarray,
    gene_names: list[str],
    evaluable_genes_set: set[str]
) -> float:
    """
    Computes the Area Under the Precision-Recall Curve (AUPRC / Average Precision)
    against a binary gold standard.

    AUPRC is preferred over AUC-ROC for imbalanced networks (e.g., 5 % density),
    where the large number of true negatives inflates ROC regardless of model quality.
    Most modern GRN benchmarks (GENIE3, SCENIC+, VIPER) report Average Precision.

    Args:
        score_adj (np.ndarray): Continuous predicted edge-score matrix (N, N).
        truth_adj (np.ndarray): Binary gold-standard adjacency matrix (N, N).
        gene_names (list[str]): Gene names corresponding to matrix indices.
        evaluable_genes_set (set[str]): Genes present in both DEG list and gold standard.

    Returns:
        float: Average Precision (area under precision-recall curve). Returns 0.0
            if fewer than 2 evaluable genes or no positive edges in gold standard.
    """
    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    if len(eval_indices) < 2:
        return 0.0

    sub_score = score_adj[np.ix_(eval_indices, eval_indices)]
    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]

    upper_mask = np.triu(np.ones_like(sub_truth, dtype=bool), k=1)
    y_score = np.nan_to_num(sub_score[upper_mask], nan=0.0, posinf=0.0, neginf=0.0)
    y_true  = sub_truth[upper_mask].astype(int)

    if y_true.sum() == 0:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def compute_precision_at_k(
    score_adj: np.ndarray,
    truth_adj: np.ndarray,
    gene_names: list[str],
    evaluable_genes_set: set[str],
    k_values: list[int] = None
) -> dict[int, float]:
    """
    Computes Precision@K for a set of K values.

    Precision@K measures what fraction of the top-K most-confident predicted
    edges are true positives. This is directly meaningful when a biologist
    plans to validate a fixed number of predictions experimentally.

    Args:
        score_adj (np.ndarray): Continuous edge-score matrix (N, N).
        truth_adj (np.ndarray): Binary gold-standard adjacency matrix (N, N).
        gene_names (list[str]): Gene names.
        evaluable_genes_set (set[str]): Evaluable gene set (DEG ∩ gold standard).
        k_values (list[int]): List of K values to evaluate. Defaults to [50, 100, 200, 500].

    Returns:
        dict[int, float]: Mapping from K to Precision@K value.
    """
    if k_values is None:
        k_values = [50, 100, 200, 500]

    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    if len(eval_indices) < 2:
        return {k: 0.0 for k in k_values}

    sub_score = score_adj[np.ix_(eval_indices, eval_indices)]
    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]

    upper_mask = np.triu(np.ones_like(sub_truth, dtype=bool), k=1)
    y_score = np.nan_to_num(sub_score[upper_mask], nan=0.0, posinf=0.0, neginf=0.0)
    y_true  = sub_truth[upper_mask].astype(int)

    sorted_idx = np.argsort(y_score)[::-1]
    results: dict[int, float] = {}
    for k in k_values:
        effective_k = min(k, len(sorted_idx))
        results[k] = float(y_true[sorted_idx[:effective_k]].mean()) if effective_k > 0 else 0.0
    return results


def print_final_summary_tables(all_results: dict[str, any]):
    """
    Prints consolidated comparative performance and topology summary tables.

    Uses Pandas to format and output two summary tables to stdout:
    1. A performance table comparing model AUC, AUPRC, Precision@K, Precision,
       Recall, and F1-score across conditions and Gold Standards.
    2. A topological table comparing node/edge counts, average degree, clustering,
       diameter, and scale-free fit (R2) for all model networks.

    Args:
        all_results (dict[str, any]): Nested dictionary structured as:
            {condition: {gs_name: {model_name: metrics_dict}, 'Topology': {model_name: topo_dict}}}.
    """
    import pandas as pd

    print(f"\n{'#'*70}")
    print(f"{' '*20} RESUMEN FINAL DE BENCHMARKS")
    print(f"{'#'*70}")

    rows = []
    for condition, gs_data in all_results.items():
        for gs_name in ["BioGRID", "STRING", "GeneMANIA"]:
            if gs_name in gs_data:
                for model_name, m in gs_data[gs_name].items():
                    # Format as mean ± std when multi-seed data is available
                    auc_str = (f"{m.get('AUC', 0):.4f} ± {m.get('AUC_std', 0):.4f}"
                               if m.get('AUC_std', 0) > 0
                               else f"{m.get('AUC', 0):.4f}")
                    auprc_str = (f"{m.get('AUPRC', 0):.4f} ± {m.get('AUPRC_std', 0):.4f}"
                                 if m.get('AUPRC_std', 0) > 0
                                 else f"{m.get('AUPRC', 0):.4f}")
                    f1_str = (f"{m.get('F1-Score', 0):.4f} ± {m.get('F1-Score_std', 0):.4f}"
                              if m.get('F1-Score_std', 0) > 0
                              else f"{m.get('F1-Score', 0):.4f}")
                    rows.append({
                        "Condición":        condition,
                        "Gold Standard":    gs_name,
                        "Modelo":           model_name,
                        "AUC (mean±std)":   auc_str,
                        "AUPRC (mean±std)": auprc_str,
                        "P@100":            f"{m.get('P@100', 0):.4f}",
                        "Precision":        f"{m['Precision']:.4f}",
                        "Recall":           f"{m['Recall']:.4f}",
                        "F1 (mean±std)":    f1_str
                    })

    
    df_perf = pd.DataFrame(rows)
    print("\n[TABLA 1: RENDIMIENTO VS GOLD STANDARDS (SUB-RED COMÚN)]")
    print(df_perf.to_string(index=False))
    
    rows_topo = []
    for condition, gs_data in all_results.items():
        topo_data = gs_data.get("Topology", {})
        for model_name, m in topo_data.items():
            rows_topo.append({
                "Condición": condition,
                "Modelo": model_name,
                "Nodes": m.get("Nodes", 0),
                "Edges": m.get("Edges", 0),
                "Avg Degree": f"{m.get('Avg Degree', 0):.2f}",
                "Clustering": f"{m.get('Clustering Coef', 0):.4f}",
                "Diameter": m.get("Diameter (LCC)", 0),
                "Scale-Free R²": f"{m.get('Scale-Free R²', 0):.4f}"
            })
            
    df_topo = pd.DataFrame(rows_topo)
    print("\n[TABLA 2: MÉTRICAS DE TOPOLOGÍA (REDES COMPLETAS)]")
    print(df_topo.to_string(index=False))
    print(f"\n{'#'*70}\n")

def print_evaluation_report(models_dict: dict[str, np.ndarray], 
                             truth_adj: np.ndarray, 
                             gene_names: list[str],
                             evaluable_genes: set[str],
                             gs_name: str,
                             condition: str):
    """
    Prints a detailed comparative evaluation report for a single gold standard.

    Compares multiple models (Pearson, Spearman, ARACNE, WGCNA, and TA-GAT) 
    against a specific reference network, reporting scale-free topology fit (R2), 
    and performance statistics (accuracy, precision, recall, F1, and predicted edge count) 
    restricted to the evaluable sub-network.

    Args:
        models_dict (dict[str, np.ndarray]): Dictionary mapping model names to their binary 
            predicted adjacency matrices of shape (num_genes, num_genes).
        truth_adj (np.ndarray): Binary gold standard adjacency matrix of shape (num_genes, num_genes).
        gene_names (list[str]): List of gene names corresponding to the indices of the adjacency matrices.
        evaluable_genes (set[str]): Set of gene names present in both the DEGs and the gold standard.
        gs_name (str): Name of the gold standard reference database (e.g., 'BioGRID').
        condition (str): The biological condition being evaluated (e.g., 'Tumor').
    """
    import networkx as nx
    
    print(f"\n==================================================")
    print(f" EVALUACIÓN DE INFERENCIA DE RED: {condition.upper()} | GS: {gs_name}")
    print(f"==================================================")
    
    G_truth = nx.from_numpy_array(truth_adj)
    r2_truth = compute_scale_free_fit(G_truth)
    
    print("\n[Ajuste Topológico Scale-Free R²]")
    print(f"  - Gold Standard ({gs_name}): {r2_truth:.4f}")
    
    for name, adj in models_dict.items():
        G_model = nx.from_numpy_array(adj)
        r2_model = compute_scale_free_fit(G_model)
        num_n = G_model.number_of_nodes()
        num_e = G_model.number_of_edges()
        print(f"  - {name}: R²={r2_model:.4f}, Nodes={num_n}, Edges={num_e}")
    
    print(f"\n[Rendimiento Fair Evaluation sobre {len(evaluable_genes)} genes comunes con {gs_name}]")
    for name, adj in models_dict.items():
        eval_metrics = evaluate_against_gold_standard(adj, truth_adj, gene_names, evaluable_genes)
        print(f"  {name}:")
        print(f"    Precision: {eval_metrics['Precision']:.4f}, Recall: {eval_metrics['Recall']:.4f}, Accuracy: {eval_metrics['Accuracy']:.4f}, F1: {eval_metrics['F1-Score']:.4f}, Valid Pred Edges: {eval_metrics['Pred_Edges']}")
    
    return

##################################################
# PHASE 3: DIAGNOSTIC CURVES & ROC VISUALIZATIONS#
##################################################

def plot_roc_curves(models_scores: dict[str, np.ndarray], 
                    truth_adj: np.ndarray, 
                    gene_names: list[str], 
                    evaluable_genes_set: set[str], 
                    condition: str, 
                    gs_name: str, 
                    dataset_name: str, 
                    save_dir: Path):
    """
    Generates and saves ROC curves comparing continuous edge scores against a gold standard.

    Calculates the True Positive Rate (TPR) and False Positive Rate (FPR) at various 
    thresholds for all models within the evaluable sub-network. Computes the Area 
    Under the Curve (AUC) and saves the comparative plot as a high-resolution PNG file.

    Args:
        models_scores (dict[str, np.ndarray]): Dictionary mapping model names to their 
            continuous predicted edge score matrices of shape (num_genes, num_genes).
        truth_adj (np.ndarray): Binary gold standard adjacency matrix of shape (num_genes, num_genes).
        gene_names (list[str]): List of gene names corresponding to the indices of the adjacency matrices.
        evaluable_genes_set (set[str]): Set of gene names present in both the DEGs and the gold standard.
        condition (str): The biological condition being evaluated (e.g., 'Tumor').
        gs_name (str): Name of the gold standard reference database.
        dataset_name (str): Accession ID of the dataset.
        save_dir (Path): Output directory path where the plot figure will be saved.
    """
    plt.figure(figsize=(10, 8))
    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    if len(eval_indices) < 2:
        return
    
    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]
    y_true = sub_truth[np.triu_indices_from(sub_truth, k=1)].astype(int)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for (name, score_adj), color in zip(models_scores.items(), colors):
        sub_score = score_adj[np.ix_(eval_indices, eval_indices)]
        y_score = sub_score[np.triu_indices_from(sub_score, k=1)]
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        try:
            auc_val = auc(fpr, tpr)
        except ValueError:
            auc_val = 0.0
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison - {dataset_name} {condition} ({gs_name})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / f"roc_{dataset_name.lower()}_{condition.lower()}_{gs_name.lower()}.png", dpi=300)
    plt.close()


##################################################
# PHASE 4: PRECISION-RECALL CURVES              #
##################################################

def plot_pr_curves(models_scores: dict[str, np.ndarray],
                   truth_adj: np.ndarray,
                   gene_names: list[str],
                   evaluable_genes_set: set[str],
                   condition: str,
                   gs_name: str,
                   dataset_name: str,
                   save_dir):
    """
    Generates and saves Precision-Recall curves comparing edge scores against a gold standard.

    Preferred over ROC curves for the highly imbalanced edge-classification task in GRN
    inference (e.g., 5 % positive rate), where AUPRC gives a more discriminative signal.
    The random-classifier baseline is plotted as a horizontal line at the positive rate.

    Args:
        models_scores (dict[str, np.ndarray]): Model names → continuous score matrices (N, N).
        truth_adj (np.ndarray): Binary gold-standard adjacency matrix (N, N).
        gene_names (list[str]): Gene names corresponding to matrix indices.
        evaluable_genes_set (set[str]): Evaluable gene set (DEG ∩ gold standard).
        condition (str): Biological condition (e.g., 'Tumor').
        gs_name (str): Gold standard name (e.g., 'STRING').
        dataset_name (str): Dataset accession or local name.
        save_dir: Output directory path for the figure.
    """
    plt.figure(figsize=(10, 8))
    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    if len(eval_indices) < 2:
        plt.close()
        return

    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]
    y_true = sub_truth[np.triu_indices_from(sub_truth, k=1)].astype(int)

    if y_true.sum() == 0:
        plt.close()
        return

    # Random-classifier baseline: horizontal line at positive rate
    baseline = float(y_true.mean())
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--',
                label=f'Random (AP = {baseline:.3f})')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for (name, score_adj), color in zip(models_scores.items(), colors):
        sub_score = score_adj[np.ix_(eval_indices, eval_indices)]
        y_score = sub_score[np.triu_indices_from(sub_score, k=1)]
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            plt.plot(rec, prec, color=color, lw=2, label=f'{name} (AP = {ap:.3f})')
        except Exception:
            pass

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name} {condition} ({gs_name})')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_dir / f"pr_{dataset_name.lower()}_{condition.lower()}_{gs_name.lower()}.png", dpi=300)
    plt.close()


##################################################
# PHASE 5: STATISTICAL SIGNIFICANCE TESTING     #
##################################################

def wilcoxon_pairwise_test(
    all_results_list: list[dict],
    metric: str = "AUC",
    gs_name: str = "STRING"
) -> dict[str, dict]:
    """
    Performs one-sided Wilcoxon signed-rank tests (TA-GAT > each baseline).

    For each baseline model, tests whether TA-GAT achieves a significantly higher
    value of 'metric' across all seed runs. Requires >= 3 seed runs; at least 5
    seeds are recommended for meaningful statistical power.

    Args:
        all_results_list (list[dict]): List of per-seed results dicts, structured as
            {gs_name: {model_name: {metric: value, ...}}}.
        metric (str): Metric key to compare (e.g., 'AUC', 'AUPRC', 'F1-Score').
        gs_name (str): Gold standard to compare against (e.g., 'STRING').

    Returns:
        dict[str, dict]: Mapping from baseline model name to
            {'statistic': float, 'p_value': float, 'tagat_mean': float, 'baseline_mean': float}.
    """
    baselines = ["Pearson", "Spearman", "ARACNE", "WGCNA"]
    tagat_vals = [r.get(gs_name, {}).get("TA-GAT", {}).get(metric, 0.0)
                  for r in all_results_list]

    results: dict[str, dict] = {}
    n_seeds = len(tagat_vals)

    for baseline in baselines:
        baseline_vals = [r.get(gs_name, {}).get(baseline, {}).get(metric, 0.0)
                         for r in all_results_list]

        entry: dict = {
            "tagat_mean":    float(np.mean(tagat_vals)),
            "baseline_mean": float(np.mean(baseline_vals)),
            "n_seeds":       n_seeds
        }

        if n_seeds < 3:
            entry["statistic"] = None
            entry["p_value"]   = None
            entry["note"]      = f"< 3 seeds — insufficient power (n={n_seeds})"
        elif all(t == b for t, b in zip(tagat_vals, baseline_vals)):
            entry["statistic"] = 0.0
            entry["p_value"]   = 1.0
            entry["note"]      = "identical values across seeds"
        else:
            try:
                stat, p = _wilcoxon(tagat_vals, baseline_vals, alternative='greater')
                entry["statistic"] = float(stat)
                entry["p_value"]   = float(p)
            except Exception as exc:
                entry["statistic"] = None
                entry["p_value"]   = None
                entry["note"]      = str(exc)

        results[baseline] = entry

    return results


def print_wilcoxon_table(wilcoxon_results: dict[str, dict], metric: str, gs_name: str, condition: str):
    """
    Prints a formatted Wilcoxon significance table comparing TA-GAT against baselines.

    Args:
        wilcoxon_results (dict): Output of wilcoxon_pairwise_test().
        metric (str): Metric used in the comparison.
        gs_name (str): Gold standard used.
        condition (str): Condition name (e.g., 'Tumor').
    """
    import pandas as pd

    rows = []
    for baseline, entry in wilcoxon_results.items():
        p_val = entry.get("p_value")
        sig = ""
        if p_val is not None:
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
        rows.append({
            "Baseline":          baseline,
            f"TA-GAT {metric}":  f"{entry['tagat_mean']:.4f}",
            f"Baseline {metric}": f"{entry['baseline_mean']:.4f}",
            "p-value":           f"{p_val:.4f}" if p_val is not None else "N/A",
            "Sig.":              sig,
            "Seeds":             entry.get("n_seeds", "?"),
            "Note":              entry.get("note", "")
        })

    df = pd.DataFrame(rows)
    print(f"\n[Wilcoxon Test: TA-GAT vs Baselines | {condition} | {gs_name} | metric={metric}]")
    print(df.to_string(index=False))
    print("  (* p<0.05  ** p<0.01  *** p<0.001  — one-sided Wilcoxon signed-rank, TA-GAT > baseline)")
