import numpy as np
import networkx as nx
from scipy.stats import linregress
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def compute_scale_free_fit(G: nx.Graph) -> float:
    """
    Calcula el ajuste (R^2) a una distribución Scale-Free (Ley de potencias).
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

def compute_network_topology_metrics(G: nx.Graph) -> dict:
    """
    Calcula un set completo de métricas topológicas para una red biológica.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # 1. Ajuste Scale-Free R2
    r2 = compute_scale_free_fit(G)
    
    # 2. Grado medio
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
    
    # 3. Coeficiente de clustering medio
    # Si la red es muy densa esto puede ser lento, pero con ~50k aristas es OK.
    avg_clustering = nx.average_clustering(G)
    
    # 4. Diámetro (sobre el componente conexo más grande para evitar infinitos)
    if num_nodes > 0 and num_edges > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        diameter = nx.diameter(G_sub)
    else:
        diameter = 0
        
    # 5. Densidad
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

def evaluate_against_gold_standard(pred_adj: np.ndarray, truth_adj: np.ndarray, gene_names: list, evaluable_genes_set: set) -> dict:
    """
    [STRICT EVALUATION]: Solo considera la sub-red formada por genes que existen
    tanto en la lista de DEGs como en el Gold Standard. Esto asegura que no estemos
    evaluando aristas donde no tenemos información de 'verdad' (truth).
    """
    # 1. Identificar índices de los genes que están en el Gold Standard
    eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
    
    if len(eval_indices) < 2:
        return {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "Pred_Edges": 0, "True_Edges": 0, "Common_Genes": 0}
    
    # 2. Extraer sub-matrices
    sub_pred = pred_adj[np.ix_(eval_indices, eval_indices)]
    sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]
    
    # 3. Solo triángulo superior
    upper_mask = np.triu(np.ones_like(sub_truth, dtype=bool), k=1)
    
    y_pred = sub_pred[upper_mask].astype(int)
    y_true = sub_truth[upper_mask].astype(int)
    
    # 4. Cálculo métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Pred_Edges": int(np.sum(y_pred)),
        "True_Edges": int(np.sum(y_true)),
        "Common_Genes": len(eval_indices)
    }

def print_final_summary_tables(all_results: dict):
    """
    Imprime tablas finales comparativas usando Pandas para mayor legibilidad.
    all_results: { 'Tumor': { 'BioGRID': { model: metrics, ... }, 'STRING': {...}, 'Topology': {...} }, 'Normal': ... }
    """
    import pandas as pd
    
    print(f"\n{'#'*70}")
    print(f"{' '*20} RESUMEN FINAL DE BENCHMARKS")
    print(f"{'#'*70}")
    
    # 1. Tabla de Rendimiento Biológico (Precision, Recall, Accuracy)
    rows = []
    for condition, gs_data in all_results.items():
        for gs_name in ["BioGRID", "STRING", "GeneMANIA"]:
            if gs_name in gs_data:
                for model_name, m in gs_data[gs_name].items():
                    rows.append({
                        "Condición": condition,
                        "Gold Standard": gs_name,
                        "Modelo": model_name,
                        "Precision": f"{m['Precision']:.4f}",
                        "Recall": f"{m['Recall']:.4f}",
                        "Accuracy": f"{m['Accuracy']:.4f}",
                        "F1": f"{m['F1-Score']:.4f}"
                    })
    
    df_perf = pd.DataFrame(rows)
    print("\n[TABLA 1: RENDIMIENTO VS GOLD STANDARDS (SUB-RED COMÚN)]")
    print(df_perf.to_string(index=False))
    
    # 2. Tabla de Topología (Métricas extendidas)
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

def print_evaluation_report(models_dict: dict, 
                            truth_adj: np.ndarray, 
                            gene_names: list,
                            evaluable_genes: set,
                            gs_name: str,
                            condition: str):


    """
    Compara las métricas topológicas y biológicas de múltiples redes (Pearson, Spearman, TA-GAT, etc.)
    frente a un Gold Standard específico.
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

def plot_roc_curves(models_scores: dict, truth_adj: np.ndarray, gene_names: list, evaluable_genes_set: set, condition: str, gs_name: str, dataset_name: str, save_dir):
    """
    Generate ROC curves for continuous predictions against the gold standard.
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

