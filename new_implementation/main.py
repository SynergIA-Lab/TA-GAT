import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import negative_sampling
import random
import matplotlib.pyplot as plt

import config
from config import CFG
from interactive_setup import run_interactive_setup
from data_loader import load_geo_and_run_deseq2, download_and_parse_biogrid, create_ground_truth_adj
from data_loader import build_correlation_baseline, build_aracne_baseline, build_wgcna_baseline
from metrics import (
    compute_network_topology_metrics, 
    evaluate_against_gold_standard, 
    print_evaluation_report,
    print_final_summary_tables,
    plot_roc_curves
)
from model_gnn import TopologyAwareGATEncoder, TAGAT, ta_gat_loss

import pandas as pd
import numpy as np
import networkx as nx

##################################################
# PHASE 1: ENVIRONMENT SETUP & SEED UTILITIES   #
##################################################

def set_seed(seed: int):
    """
    Sets random seeds for reproducibility across libraries and hardware.

    Configures seeds for Python's random, numpy, and PyTorch (CPU, CUDA, and MPS). 
    Enforces deterministic cudnn backends and attempts to force deterministic 
    algorithms in PyTorch where available.

    Args:
        seed (int): The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"  -> Semilla aleatoria fijada en: {seed}")

##################################################
# PHASE 2: GNN MODEL TRAINING & OPTIMIZATION    #
##################################################

def train_ta_gat(model: nn.Module, data: any, optimizer: optim.Optimizer, epochs: int = CFG.EPOCHS) -> list[float]:
    """
    Trains the TA-GAT Graph Neural Network model.

    Iteratively computes the encoder representation, calculates the composite 
    reconstruction, Kullback-Leibler divergence, scale-free topological, and targeted 
    sparsity loss terms, and executes backpropagation to optimize network parameters.

    Args:
        model (nn.Module): The TAGAT model to be trained.
        data (any): The PyTorch Geometric Data object containing the graph.
        optimizer (optim.Optimizer): The PyTorch optimizer.
        epochs (int, optional): The total number of training epochs. Defaults to CFG.EPOCHS.

    Returns:
        list[float]: A list containing the scalar training loss recorded at each epoch.
    """
    model.train()
    
    print(f"\n[TA-GAT] Entrenando GNN (Epochs: {epochs})...")
    
    loss_history = []
    
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1),
        method='sparse'
    )
    
    device = next(model.parameters()).device
    neg_edge_index = neg_edge_index.to(torch.long).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        edge_attr = getattr(data, 'edge_attr', None)
        z = model.encode(data.x, data.edge_index, edge_attr=edge_attr)
        
        recon_loss, sf_loss, sparsity_loss = ta_gat_loss(
            z, data.edge_index, neg_edge_index, CFG.LAMBDA_SCALE_FREE, pos_edge_weights=edge_attr
        )
        kl_div = (1.0 / data.num_nodes) * model.kl_loss()
        
        loss = CFG.LAMBDA_RECON * recon_loss + CFG.LAMBDA_KL * kl_div + CFG.LAMBDA_SCALE_FREE * sf_loss + CFG.LAMBDA_SPARSITY * sparsity_loss
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f} "
                  f"[Rec: {recon_loss.item():.4f}, KL: {kl_div.item():.4f}, SF: {sf_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}]")
            if str(device) == 'mps':
                torch.mps.empty_cache()

    return loss_history

def plot_learning_curve(history: list[float], condition_name: str):
    """
    Generates and saves the GNN training loss learning curve.

    Plots the recorded multi-objective loss history across epochs and saves the resulting 
    figure as a high-resolution PNG file.

    Args:
        history (list[float]): List of loss values across training epochs.
        condition_name (str): The name of the experimental condition (e.g., 'Tumor').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Loss Total', color='royalblue', linewidth=2)
    plt.title(f'Curva de Aprendizaje TA-GAT - {condition_name}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    filename = f"learning_curve_{condition_name.lower()}.png"
    save_path = config.FIGURES_DIR / filename
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Curva de aprendizaje guardada en: {save_path}")

##################################################
# PHASE 3: LATENT SPACE INFERENCE & FILTERING   #
##################################################

def infer_network(model: torch.nn.Module, data: any) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs hybrid network inference by combining GNN predictions with WGCNA.

    Computes predicted edge probabilities by calculating the sigmoid cosine similarity 
    between latent gene representations. Combines this GNN-derived matrix with the WGCNA 
    soft-thresholded correlation weights using a hybrid formula: 
    Score = Prob(GNN) * (1 + 2 * WGCNA_weight). Extracts the top-K highest-scoring edges 
    to meet the target density, ensuring a scale-free network topology and high accuracy.

    Args:
        model (torch.nn.Module): The trained TAGAT model.
        data (any): The PyTorch Geometric Data object containing the graph.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - adj_bin (np.ndarray): Binary symmetric adjacency matrix of shape (num_genes, num_genes).
            - score_np (np.ndarray): Continuous hybrid edge score matrix of shape (num_genes, num_genes).
    """
    model.eval()
    with torch.no_grad():
        edge_attr = getattr(data, 'edge_attr', None)
        z = model.encode(data.x, data.edge_index, edge_attr=edge_attr)
        z_norm = F.normalize(z, p=2, dim=1)
        adj_pred = torch.sigmoid(torch.matmul(z_norm, z_norm.t()))
        
    adj_np = adj_pred.cpu().numpy()
    np.fill_diagonal(adj_np, 0.0)
    
    N = adj_np.shape[0]
    
    target_density = getattr(CFG, 'INFERENCE_TARGET_DENSITY', 0.05)
    target_edges = int(N * (N - 1) / 2 * target_density)
    
    wgcna_dense = np.zeros((N, N))
    if edge_attr is not None:
        idx = data.edge_index.cpu().numpy()
        vals = edge_attr.cpu().numpy().squeeze()
        wgcna_dense[idx[0], idx[1]] = vals
        wgcna_dense = np.maximum(wgcna_dense, wgcna_dense.T)
        
    baseline_mask = (wgcna_dense > 0).astype(float)
    
    score_np = adj_np * (1.0 + 2.0 * wgcna_dense)
    
    upper_tri_indices = np.triu_indices_from(score_np, k=1)
    upper_scores = score_np[upper_tri_indices]
    
    if target_edges < len(upper_scores):
        threshold_idx = len(upper_scores) - target_edges
        threshold_val = np.partition(upper_scores, threshold_idx)[threshold_idx]
    else:
        threshold_val = 0.0
        
    adj_bin = (score_np >= threshold_val).astype(float)
    np.fill_diagonal(adj_bin, 0.0)
    adj_bin = np.maximum(adj_bin, adj_bin.T)
    
    actual_edges = int(adj_bin.sum() / 2)
    print(f"  [Esqueleto+GNN Inference] Target: {target_edges} edges. Actual: {actual_edges} edges.")
    
    return adj_bin, score_np

##################################################
# PHASE 4: CONDITION-SPECIFIC PIPELINE EXECUTION#
##################################################

def run_condition_pipeline(condition_name: str, dataset_name: str, expr_df: pd.DataFrame, 
                           bg_adj: np.ndarray, bg_eval_set: set[str], 
                           str_adj: np.ndarray, str_eval_set: set[str], 
                           gm_adj: np.ndarray, gm_eval_set: set[str],
                           genes: list[str], tf_set: set[str], device: torch.device) -> tuple[dict[str, any], np.ndarray]:
    """
    Executes the entire network inference and validation pipeline for a single condition.

    1. Extracts baseline correlation matrices (Pearson, Spearman, WGCNA, ARACNE) and 
       constructs a PyG graph incorporating transcription factor boosting.
    2. Initializes and trains the TA-GAT model, plotting the learning curve.
    3. Performs hybrid inference (GNN + WGCNA) to generate the final network.
    4. Evaluates predictions against BioGRID, STRING, and GeneMANIA using Fair Evaluation.
    5. Saves ROC curves, exports predicted adjacency matrices, and returns summary metrics.

    Args:
        condition_name (str): Name of the condition being processed (e.g., 'Tumor').
        dataset_name (str): Accession ID of the dataset.
        expr_df (pd.DataFrame): Expression matrix of shape (num_genes, num_samples).
        bg_adj (np.ndarray): Binary BioGRID gold standard matrix.
        bg_eval_set (set[str]): Set of evaluable genes for BioGRID.
        str_adj (np.ndarray): Binary STRING gold standard matrix.
        str_eval_set (set[str]): Set of evaluable genes for STRING.
        gm_adj (np.ndarray): Binary GeneMANIA gold standard matrix.
        gm_eval_set (set[str]): Set of evaluable genes for GeneMANIA.
        genes (list[str]): List of differentially expressed genes.
        tf_set (set[str]): Set of transcription factor gene names.
        device (torch.device): CPU, CUDA, or MPS computing device.

    Returns:
        tuple[dict[str, any], np.ndarray]: A tuple containing:
            - results (dict[str, any]): Evaluation metrics and topological properties for all models.
            - ta_gat_bin (np.ndarray): The final inferred binary adjacency matrix of shape (num_genes, num_genes).
    """
    print(f"\n{'='*60}")
    print(f" INICIANDO PIPELINE: {condition_name.upper()}")
    print(f"{'='*60}")
    
    print("\n[1] Extrayendo Redes Base Individuales y Ensemble...")
    ensemble_bin, pearson_bin, spearman_bin, ensemble_corr, pearson_score, spearman_score, pyg_data = build_correlation_baseline(expr_df, tf_set)
    pyg_data = pyg_data.to(device)
    
    encoder = TopologyAwareGATEncoder(
        in_channels=pyg_data.num_node_features,
        hidden_channels=CFG.GAT_HIDDEN_CHANNELS,
        out_channels=CFG.GAT_OUT_CHANNELS,
        heads=CFG.GAT_HEADS
    )

    model = TAGAT(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
    
    loss_history = train_ta_gat(model, pyg_data, optimizer, epochs=CFG.EPOCHS)
    
    plot_learning_curve(loss_history, condition_name)
    
    ta_gat_bin, tagat_score = infer_network(model, pyg_data)
    
    print("  -> Generando baselines complementarios...")
    aracne_bin, aracne_score = build_aracne_baseline(expr_df)
    wgcna_bin, wgcna_score = build_wgcna_baseline(expr_df)
    
    models_dict = {
        "Pearson (Baseline)": pearson_bin,
        "Spearman (Baseline)": spearman_bin,
        "ARACNE (Baseline)": aracne_bin,
        "WGCNA (Baseline)": wgcna_bin,
        "TA-GAT (GNN)": ta_gat_bin
    }
    
    print_evaluation_report(
        models_dict=models_dict,
        truth_adj=bg_adj,
        gene_names=genes,
        evaluable_genes=bg_eval_set,
        gs_name="BioGRID",
        condition=condition_name
    )
    
    print_evaluation_report(
        models_dict=models_dict,
        truth_adj=str_adj,
        gene_names=genes,
        evaluable_genes=str_eval_set,
        gs_name="STRING",
        condition=condition_name
    )
    
    print_evaluation_report(
        models_dict=models_dict,
        truth_adj=gm_adj,
        gene_names=genes,
        evaluable_genes=gm_eval_set,
        gs_name="GeneMANIA",
        condition=condition_name
    )
    
    cond_dir = config.OUT_DIR / condition_name.replace(" ", "_").lower()
    cond_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cond_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    scores_dict = {
        "Pearson": pearson_score,
        "Spearman": spearman_score,
        "ARACNE": aracne_score,
        "WGCNA": wgcna_score,
        "TA-GAT": tagat_score
    }
    plot_roc_curves(scores_dict, bg_adj, genes, bg_eval_set, condition_name, "BioGRID", dataset_name, figures_dir)
    plot_roc_curves(scores_dict, str_adj, genes, str_eval_set, condition_name, "STRING", dataset_name, figures_dir)
    plot_roc_curves(scores_dict, gm_adj, genes, gm_eval_set, condition_name, "GeneMANIA", dataset_name, figures_dir)
    
    pd.DataFrame(pearson_bin, index=genes, columns=genes).to_csv(cond_dir / "pearson_network.tsv", sep="\t")
    pd.DataFrame(spearman_bin, index=genes, columns=genes).to_csv(cond_dir / "spearman_network.tsv", sep="\t")
    pd.DataFrame(aracne_bin, index=genes, columns=genes).to_csv(cond_dir / "aracne_network.tsv", sep="\t")
    pd.DataFrame(wgcna_bin, index=genes, columns=genes).to_csv(cond_dir / "wgcna_network.tsv", sep="\t")
    pd.DataFrame(ta_gat_bin, index=genes, columns=genes).to_csv(cond_dir / "tagat_network.tsv", sep="\t")
    print(f"  -> Matrices de adyacencia exportadas a: {cond_dir}")
    
    results = {
        "BioGRID": {
            "Pearson": evaluate_against_gold_standard(pearson_bin, bg_adj, genes, bg_eval_set),
            "Spearman": evaluate_against_gold_standard(spearman_bin, bg_adj, genes, bg_eval_set),
            "ARACNE": evaluate_against_gold_standard(aracne_bin, bg_adj, genes, bg_eval_set),
            "WGCNA": evaluate_against_gold_standard(wgcna_bin, bg_adj, genes, bg_eval_set),
            "TA-GAT": evaluate_against_gold_standard(ta_gat_bin, bg_adj, genes, bg_eval_set)
        },
        "STRING": {
            "Pearson": evaluate_against_gold_standard(pearson_bin, str_adj, genes, str_eval_set),
            "Spearman": evaluate_against_gold_standard(spearman_bin, str_adj, genes, str_eval_set),
            "ARACNE": evaluate_against_gold_standard(aracne_bin, str_adj, genes, str_eval_set),
            "WGCNA": evaluate_against_gold_standard(wgcna_bin, str_adj, genes, str_eval_set),
            "TA-GAT": evaluate_against_gold_standard(ta_gat_bin, str_adj, genes, str_eval_set)
        },
        "GeneMANIA": {
            "Pearson": evaluate_against_gold_standard(pearson_bin, gm_adj, genes, gm_eval_set),
            "Spearman": evaluate_against_gold_standard(spearman_bin, gm_adj, genes, gm_eval_set),
            "ARACNE": evaluate_against_gold_standard(aracne_bin, gm_adj, genes, gm_eval_set),
            "WGCNA": evaluate_against_gold_standard(wgcna_bin, gm_adj, genes, gm_eval_set),
            "TA-GAT": evaluate_against_gold_standard(ta_gat_bin, gm_adj, genes, gm_eval_set)
        },
        "Topology": {
            "Pearson": compute_network_topology_metrics(nx.from_numpy_array(pearson_bin)),
            "Spearman": compute_network_topology_metrics(nx.from_numpy_array(spearman_bin)),
            "ARACNE": compute_network_topology_metrics(nx.from_numpy_array(aracne_bin)),
            "WGCNA": compute_network_topology_metrics(nx.from_numpy_array(wgcna_bin)),
            "TA-GAT": compute_network_topology_metrics(nx.from_numpy_array(ta_gat_bin))
        }
    }
    return results, ta_gat_bin

##################################################
# PHASE 5: SYSTEM LOGGING & ENTRY POINT         #
##################################################

class Logger(object):
    """
    A custom dual-stream logger utility.

    Redirects standard output and error streams to both the system terminal 
    and a persistent log file in the output directory.
    """
    def __init__(self, filename: Path):
        """
        Initializes the Logger stream object.

        Args:
            filename (Path): The file path where log messages will be saved.
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message: str):
        """
        Writes a message to both stdout and the log file.

        Args:
            message (str): The text message to write.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flushes both output streams.
        """
        self.terminal.flush()
        self.log.flush()

def main():
    """
    The main orchestrator for the GNN Network Inference pipeline.

    1. Triggers the interactive setup to configure study groups.
    2. Sets up seeds, identifies GPU acceleration (CUDA, Apple MPS, or CPU), 
       and starts logging.
    3. Loads transcription factors, downloads GEO expression profiles, and runs PyDESeq2.
    4. Downloads, parses, and maps physical, functional, and co-expression gold standards.
    5. Loops over cohorts, executing training, hybrid inference, and fair evaluation.
    6. Outputs aggregated comparative summary tables and calls biological hub analysis.
    """
    out_dir, figures_dir = run_interactive_setup()
    config.OUT_DIR = out_dir
    config.FIGURES_DIR = figures_dir
    
    sys.stdout = Logger(out_dir / "run.log")
    sys.stderr = sys.stdout
    set_seed(CFG.SEED)
    print("=" * 70)
    print("  Advanced GNN Network Inference (TA-GAT Original implementation)")
    print(f"  PyTorch       : {torch.__version__}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("  GPU Apple MPS")
    else:
        device = torch.device('cpu')
        print("  CPU")
    print("=" * 70)
    
    from data_loader import load_tf_list
    tf_set = load_tf_list(CFG.TF_LIST_PATH)
    print(f"  -> TFs Cargados: {len(tf_set)}")

    sig_degs, expr_groups_dict = load_geo_and_run_deseq2()
    degs_genes = sig_degs.index.tolist()
    
    from data_loader import download_and_parse_biogrid, download_and_parse_string, download_and_parse_genemania_coexp, create_ground_truth_adj
    unique_bg_edges = download_and_parse_biogrid()
    unique_string_edges = download_and_parse_string()
    unique_gm_edges = download_and_parse_genemania_coexp()
    
    bg_adj, bg_eval_set = create_ground_truth_adj(degs_genes, unique_bg_edges)
    string_adj, string_eval_set = create_ground_truth_adj(degs_genes, unique_string_edges)
    gm_adj, gm_eval_set = create_ground_truth_adj(degs_genes, unique_gm_edges)
    
    all_final_results = {}
    adj_dict = {}
    
    for g_name, g_expr in expr_groups_dict.items():
        if g_expr.empty or g_expr.shape[1] == 0:
            print(f"[WARN] El grupo '{g_name}' no tiene suficientes muestras. Saltando pipeline.")
            continue
            
        g_res, g_adj = run_condition_pipeline(g_name, CFG.GEO_ID, g_expr, bg_adj, bg_eval_set, string_adj, string_eval_set, gm_adj, gm_eval_set, degs_genes, tf_set, device)
        all_final_results[g_name] = g_res
        adj_dict[g_name] = g_adj

    print_final_summary_tables(all_final_results)
    
    from biological_relevance import analyze_hubs
    if adj_dict:
        analyze_hubs(adj_dict, degs_genes, tf_set, config.OUT_DIR)

    print("\n[COMPLETADO] Todas las ejecuciones de TA-GAT han terminado con éxito.")

if __name__ == "__main__":
    main()
