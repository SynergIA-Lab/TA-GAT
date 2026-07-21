import os
import sys
from pathlib import Path
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
    compute_auc_against_gs,
    compute_auprc_against_gs,
    compute_precision_at_k,
    print_evaluation_report,
    print_final_summary_tables,
    plot_roc_curves,
    plot_pr_curves,
    wilcoxon_pairwise_test,
    print_wilcoxon_table
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
    Trains the TA-GAT Graph Neural Network model with early stopping.

    Iteratively computes the encoder representation, calculates the composite 
    reconstruction, Kullback-Leibler divergence, scale-free topological, and targeted 
    sparsity loss terms, and executes backpropagation to optimize network parameters.
    Training halts early when the total loss does not improve by at least MIN_DELTA
    for PATIENCE consecutive epochs.

    Args:
        model (nn.Module): The TAGAT model to be trained.
        data (any): The PyTorch Geometric Data object containing the graph.
        optimizer (optim.Optimizer): The PyTorch optimizer.
        epochs (int, optional): Maximum number of training epochs. Defaults to CFG.EPOCHS.

    Returns:
        list[float]: A list containing the scalar training loss recorded at each epoch.
    """
    model.train()
    print(f"\n[TA-GAT] Entrenando GNN (Max Epochs: {epochs}, Patience: {CFG.PATIENCE})...")

    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    device = next(model.parameters()).device

    # When DYNAMIC_NEG_SAMPLING=False, sample once for speed; otherwise re-sample each epoch
    _static_neg = None
    if not getattr(CFG, 'DYNAMIC_NEG_SAMPLING', True):
        _static_neg = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
            method='sparse'
        ).to(torch.long).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Dynamic negative sampling: fresh negatives each epoch for better generalisation
        if getattr(CFG, 'DYNAMIC_NEG_SAMPLING', True):
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.edge_index.size(1),
                method='sparse'
            ).to(torch.long).to(device)
        else:
            neg_edge_index = _static_neg

        edge_attr = getattr(data, 'edge_attr', None)
        z = model.encode(data.x, data.edge_index, edge_attr=edge_attr)

        recon_loss, sf_loss, sparsity_loss = ta_gat_loss(
            z, data.edge_index, neg_edge_index, CFG.LAMBDA_SCALE_FREE, pos_edge_weights=edge_attr
        )
        kl_div = (1.0 / data.num_nodes) * model.kl_loss()

        loss = (CFG.LAMBDA_RECON * recon_loss
                + CFG.LAMBDA_KL * kl_div
                + CFG.LAMBDA_SCALE_FREE * sf_loss
                + CFG.LAMBDA_SPARSITY * sparsity_loss)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        # ── Early stopping ───────────────────────────────────────────────────
        if current_loss < best_loss - CFG.MIN_DELTA:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CFG.PATIENCE:
            print(f"  [Early Stop] Epoch {epoch:03d} — sin mejora en {CFG.PATIENCE} epochs consecutivos.")
            break
        # ────────────────────────────────────────────────────────────────────

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:03d}/{epochs} | Loss: {current_loss:.4f} "
                  f"[Rec: {recon_loss.item():.4f}, KL: {kl_div.item():.4f}, "
                  f"SF: {sf_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}]")
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
    Performs pure GNN network inference from latent cosine-similarity scores.

    FIX (inference): The previous implementation multiplied GNN predictions by WGCNA
    edge weights (Score = GNN * (1 + 2*WGCNA)), which structurally prevented the model
    from recovering edges that WGCNA had suppressed and set an upper bound on Recall
    equal to WGCNA's own Recall. The inference now uses the raw GNN cosine-similarity
    score matrix exclusively.

    Computes predicted edge probabilities as sigmoid(cosine_similarity(z_i, z_j)),
    exactly matching the metric used during training (consistent train/infer pipeline).
    Extracts the top-K highest-scoring edges to meet the target density.

    Args:
        model (torch.nn.Module): The trained TAGAT model.
        data (any): The PyTorch Geometric Data object containing the graph.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - adj_bin (np.ndarray): Binary symmetric adjacency matrix of shape (num_genes, num_genes).
            - score_np (np.ndarray): Continuous GNN cosine-similarity score matrix of shape (num_genes, num_genes).
    """
    model.eval()
    with torch.no_grad():
        edge_attr = getattr(data, 'edge_attr', None)
        z = model.encode(data.x, data.edge_index, edge_attr=edge_attr)
        z_norm = F.normalize(z, p=2, dim=1)
        adj_pred = torch.sigmoid(torch.matmul(z_norm, z_norm.t()))

    # FIX: Pure GNN scores — no WGCNA multiplicative boost
    score_np = adj_pred.cpu().numpy()
    np.fill_diagonal(score_np, 0.0)

    N = score_np.shape[0]
    target_density = getattr(CFG, 'INFERENCE_TARGET_DENSITY', 0.05)
    target_edges   = int(N * (N - 1) / 2 * target_density)

    upper_tri_indices = np.triu_indices_from(score_np, k=1)
    upper_scores      = score_np[upper_tri_indices]

    if target_edges < len(upper_scores):
        threshold_idx = len(upper_scores) - target_edges
        threshold_val = np.partition(upper_scores, threshold_idx)[threshold_idx]
    else:
        threshold_val = 0.0

    adj_bin = (score_np >= threshold_val).astype(float)
    np.fill_diagonal(adj_bin, 0.0)
    adj_bin = np.maximum(adj_bin, adj_bin.T)

    actual_edges = int(adj_bin.sum() / 2)
    print(f"  [GNN Inference] Target: {target_edges} edges. Actual: {actual_edges} edges.")

    return adj_bin, score_np

##################################################
# PHASE 4: CONDITION-SPECIFIC PIPELINE EXECUTION#
##################################################


def average_multi_seed_results(all_results_list: list[dict]) -> dict:
    """
    Aggregates results from multiple seed runs into mean ± std per metric.

    For each gold-standard / model combination, computes the arithmetic mean and
    standard deviation of AUC, Precision, Recall, F1-Score, and Accuracy across all
    seed runs. Topology metrics are taken from the first run (they are structurally
    seed-independent since the adjacency is binarized by top-K deterministically).

    Args:
        all_results_list (list[dict]): List of results dicts, one per seed run,
            each structured as {gs_name: {model_name: metrics_dict}}.

    Returns:
        dict: Averaged results dict with the same structure, augmented with
            '<metric>_std' keys for every numeric metric.
    """
    avg = {}
    for gs_name in ["BioGRID", "STRING", "GeneMANIA"]:
        avg[gs_name] = {}
        # Collect all model names seen across runs
        model_names = set()
        for r in all_results_list:
            model_names.update(r.get(gs_name, {}).keys())
        for model_name in model_names:
            metrics_agg = {}
            # Aggregate all numeric performance metrics dynamically (handles AUPRC, P@K, etc.)
            first_run = all_results_list[0].get(gs_name, {}).get(model_name, {})
            numeric_keys = [
                k for k, v in first_run.items()
                if isinstance(v, (int, float))
                and k not in ("Pred_Edges", "True_Edges", "Common_Genes")
            ]
            for key in numeric_keys:
                vals = [
                    r.get(gs_name, {}).get(model_name, {}).get(key, 0.0)
                    for r in all_results_list
                ]
                metrics_agg[key]           = float(np.mean(vals))
                metrics_agg[f"{key}_std"]  = float(np.std(vals))
            # Non-aggregated metadata from first run
            metrics_agg["Pred_Edges"]   = first_run.get("Pred_Edges", 0)
            metrics_agg["True_Edges"]   = first_run.get("True_Edges", 0)
            metrics_agg["Common_Genes"] = first_run.get("Common_Genes", 0)
            avg[gs_name][model_name] = metrics_agg
    # Topology is deterministic given the same adjacency binarization — use seed 0
    avg["Topology"] = all_results_list[0].get("Topology", {})
    return avg


def run_multi_seed_pipeline(
    condition_name: str, dataset_name: str, expr_df: pd.DataFrame,
    bg_adj: np.ndarray, bg_eval_set: set, str_adj: np.ndarray, str_eval_set: set,
    gm_adj: np.ndarray, gm_eval_set: set, genes: list, tf_set: set,
    device: torch.device,
    sig_degs: pd.DataFrame = None
) -> tuple[dict, np.ndarray]:
    """
    Runs the full condition pipeline multiple times with different random seeds and
    returns averaged metrics with standard deviations and an ensemble-consensus network.

    For each seed, the GNN is re-initialized and retrained. After all seeds:
    - Metrics are averaged with mean ± std via average_multi_seed_results().
    - Score matrices from all seeds are averaged to build an ensemble-consensus network
      (more stable than picking the single best-seed network).
    - Wilcoxon signed-rank tests are run for AUC and AUPRC (TA-GAT vs each baseline).

    Args:
        condition_name (str): Name of the condition being processed (e.g., 'Tumor').
        dataset_name (str): Accession ID or local name of the dataset.
        expr_df (pd.DataFrame): Expression matrix of shape (num_genes, num_samples).
        bg_adj, bg_eval_set: BioGRID gold standard matrix and evaluable gene set.
        str_adj, str_eval_set: STRING gold standard matrix and evaluable gene set.
        gm_adj, gm_eval_set: GeneMANIA gold standard matrix and evaluable gene set.
        genes (list[str]): List of differentially expressed gene names.
        tf_set (set[str]): Set of transcription factor gene names.
        device (torch.device): Computing device.
        sig_degs (pd.DataFrame, optional): PyDESeq2 results (log2FC, padj) for DESeq2 node features.

    Returns:
        tuple[dict, np.ndarray]:
            - Averaged results dict with mean ± std metrics.
            - Ensemble-consensus adjacency matrix binarized from averaged score matrices.
    """
    # FIX: FAST_MODE — single-seed development run (~20 min on M4 vs hours)
    seeds = [123] if getattr(CFG, 'FAST_MODE', False) else getattr(CFG, 'N_SEEDS', [123])
    mode_tag = " [FAST_MODE — 1 seed]" if getattr(CFG, 'FAST_MODE', False) else ""
    print(f"\n[Multi-Seed] Condición '{condition_name}' — {len(seeds)} seeds: {seeds}{mode_tag}")

    all_results: list[dict] = []
    all_scores: list[np.ndarray] = []   # tagat_score per seed for ensemble voting

    for i, seed in enumerate(seeds):
        print(f"\n  ===== SEED {seed} ({i+1}/{len(seeds)}) =====")
        set_seed(seed)

        seed_results, seed_adj, seed_score = run_condition_pipeline(
            condition_name, dataset_name, expr_df,
            bg_adj, bg_eval_set, str_adj, str_eval_set, gm_adj, gm_eval_set,
            genes, tf_set, device, sig_degs=sig_degs
        )
        all_results.append(seed_results)
        all_scores.append(seed_score)

        str_auc   = seed_results.get("STRING", {}).get("TA-GAT", {}).get("AUC",   0.0)
        str_auprc = seed_results.get("STRING", {}).get("TA-GAT", {}).get("AUPRC", 0.0)
        print(f"    → Seed {seed}: AUC={str_auc:.4f}, AUPRC={str_auprc:.4f} (STRING)")

    averaged = average_multi_seed_results(all_results)

    # ── Ensemble-consensus network: average score matrices → binarise ──────────────
    # This is more stable than selecting the single best-seed network, which
    # reflects only one random initialisation rather than the consensus signal.
    avg_score = np.mean(all_scores, axis=0)
    N = avg_score.shape[0]
    target_density  = getattr(CFG, 'INFERENCE_TARGET_DENSITY', 0.05)
    target_edges    = int(N * (N - 1) / 2 * target_density)
    upper_scores    = avg_score[np.triu_indices(N, k=1)]
    if target_edges < len(upper_scores):
        thr_idx   = len(upper_scores) - target_edges
        thr_val   = np.partition(upper_scores, thr_idx)[thr_idx]
    else:
        thr_val   = 0.0
    ensemble_adj = (avg_score >= thr_val).astype(float)
    np.fill_diagonal(ensemble_adj, 0.0)
    ensemble_adj = np.maximum(ensemble_adj, ensemble_adj.T)
    actual_e     = int(ensemble_adj.sum() / 2)
    print(f"\n[Ensemble] Red de consenso ({len(seeds)} seeds): {actual_e} aristas (objetivo: {target_edges}).")

    # ── Wilcoxon signed-rank tests (TA-GAT vs baselines) ──────────────────────
    if len(all_results) >= 2:
        for metric in ["AUC", "AUPRC"]:
            for gs in ["STRING", "BioGRID"]:
                wx = wilcoxon_pairwise_test(all_results, metric=metric, gs_name=gs)
                print_wilcoxon_table(wx, metric=metric, gs_name=gs, condition=condition_name)
    else:
        print("[Wilcoxon] Omitido: se requieren >= 2 seeds para el test ("
              f"n={len(all_results)} seed(s)).")

    best_tagat_auc = averaged.get("STRING", {}).get("TA-GAT", {}).get("AUC", 0.0)
    print(f"\n[Multi-Seed] '{condition_name}' completado. AUC STRING TA-GAT (mean): {best_tagat_auc:.4f}")
    return averaged, ensemble_adj



def run_condition_pipeline(condition_name: str, dataset_name: str, expr_df: pd.DataFrame,
                           bg_adj: np.ndarray, bg_eval_set: set[str],
                           str_adj: np.ndarray, str_eval_set: set[str],
                           gm_adj: np.ndarray, gm_eval_set: set[str],
                           genes: list[str], tf_set: set[str], device: torch.device,
                           sig_degs: pd.DataFrame = None) -> tuple[dict[str, any], np.ndarray, np.ndarray]:
    """
    Executes the entire network inference and validation pipeline for a single condition.

    1. Extracts baseline correlation matrices (Pearson, Spearman, WGCNA, ARACNE) and
       constructs a PyG graph incorporating transcription factor boosting and optional
       DESeq2 node features (log2FC, -log10(padj)).
    2. Initializes and trains the TA-GAT model with attention dropout, plotting the learning curve.
    3. Performs pure GNN inference to generate the final network.
    4. Evaluates predictions against BioGRID, STRING, and GeneMANIA using Fair Evaluation.
       Computes AUC-ROC, AUPRC, and Precision@K for each model × gold standard.
    5. Saves ROC curves, Precision-Recall curves, exports predicted adjacency matrices.

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
        sig_degs (pd.DataFrame, optional): PyDESeq2 results DataFrame (indexed by gene)
            with 'log2FoldChange' and 'padj' columns. Used as additional GNN node features.

    Returns:
        tuple[dict[str, any], np.ndarray, np.ndarray]: A tuple containing:
            - results (dict[str, any]): Evaluation metrics and topological properties for all models.
            - ta_gat_bin (np.ndarray): The final inferred binary adjacency matrix (num_genes, num_genes).
            - tagat_score (np.ndarray): Continuous GNN score matrix (num_genes, num_genes) for ensemble.
    """
    print(f"\n{'='*60}")
    print(f" INICIANDO PIPELINE: {condition_name.upper()}")
    print(f"{'='*60}")
    
    print("\n[1] Extrayendo Redes Base Individuales y Ensemble...")
    ensemble_bin, pearson_bin, spearman_bin, ensemble_corr, pearson_score, spearman_score, pyg_data = \
        build_correlation_baseline(expr_df, tf_set, degs_df=sig_degs)
    pyg_data = pyg_data.to(device)
    
    encoder = TopologyAwareGATEncoder(
        in_channels=pyg_data.num_node_features,
        hidden_channels=CFG.GAT_HIDDEN_CHANNELS,
        out_channels=CFG.GAT_OUT_CHANNELS,
        heads=CFG.GAT_HEADS,
        dropout=getattr(CFG, 'GAT_DROPOUT', 0.0)
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
    # Precision-Recall curves (more informative than ROC for imbalanced GRN task)
    plot_pr_curves(scores_dict, bg_adj, genes, bg_eval_set, condition_name, "BioGRID", dataset_name, figures_dir)
    plot_pr_curves(scores_dict, str_adj, genes, str_eval_set, condition_name, "STRING", dataset_name, figures_dir)
    plot_pr_curves(scores_dict, gm_adj, genes, gm_eval_set, condition_name, "GeneMANIA", dataset_name, figures_dir)
    
    pd.DataFrame(pearson_bin, index=genes, columns=genes).to_csv(cond_dir / "pearson_network.tsv", sep="\t")
    pd.DataFrame(spearman_bin, index=genes, columns=genes).to_csv(cond_dir / "spearman_network.tsv", sep="\t")
    pd.DataFrame(aracne_bin, index=genes, columns=genes).to_csv(cond_dir / "aracne_network.tsv", sep="\t")
    pd.DataFrame(wgcna_bin, index=genes, columns=genes).to_csv(cond_dir / "wgcna_network.tsv", sep="\t")
    pd.DataFrame(ta_gat_bin, index=genes, columns=genes).to_csv(cond_dir / "tagat_network.tsv", sep="\t")
    print(f"  -> Matrices de adyacencia exportadas a: {cond_dir}")
    
    results = {
        "BioGRID": {
            "Pearson":  evaluate_against_gold_standard(pearson_bin,  bg_adj,  genes, bg_eval_set),
            "Spearman": evaluate_against_gold_standard(spearman_bin, bg_adj,  genes, bg_eval_set),
            "ARACNE":   evaluate_against_gold_standard(aracne_bin,   bg_adj,  genes, bg_eval_set),
            "WGCNA":    evaluate_against_gold_standard(wgcna_bin,    bg_adj,  genes, bg_eval_set),
            "TA-GAT":   evaluate_against_gold_standard(ta_gat_bin,   bg_adj,  genes, bg_eval_set)
        },
        "STRING": {
            "Pearson":  evaluate_against_gold_standard(pearson_bin,  str_adj, genes, str_eval_set),
            "Spearman": evaluate_against_gold_standard(spearman_bin, str_adj, genes, str_eval_set),
            "ARACNE":   evaluate_against_gold_standard(aracne_bin,   str_adj, genes, str_eval_set),
            "WGCNA":    evaluate_against_gold_standard(wgcna_bin,    str_adj, genes, str_eval_set),
            "TA-GAT":   evaluate_against_gold_standard(ta_gat_bin,   str_adj, genes, str_eval_set)
        },
        "GeneMANIA": {
            "Pearson":  evaluate_against_gold_standard(pearson_bin,  gm_adj,  genes, gm_eval_set),
            "Spearman": evaluate_against_gold_standard(spearman_bin, gm_adj,  genes, gm_eval_set),
            "ARACNE":   evaluate_against_gold_standard(aracne_bin,   gm_adj,  genes, gm_eval_set),
            "WGCNA":    evaluate_against_gold_standard(wgcna_bin,    gm_adj,  genes, gm_eval_set),
            "TA-GAT":   evaluate_against_gold_standard(ta_gat_bin,   gm_adj,  genes, gm_eval_set)
        },
        "Topology": {
            "Pearson":  compute_network_topology_metrics(nx.from_numpy_array(pearson_bin)),
            "Spearman": compute_network_topology_metrics(nx.from_numpy_array(spearman_bin)),
            "ARACNE":   compute_network_topology_metrics(nx.from_numpy_array(aracne_bin)),
            "WGCNA":    compute_network_topology_metrics(nx.from_numpy_array(wgcna_bin)),
            "TA-GAT":   compute_network_topology_metrics(nx.from_numpy_array(ta_gat_bin))
        }
    }

    # Add AUC-ROC, AUPRC, and Precision@K to each model × gold-standard entry
    pak_values = getattr(CFG, 'PRECISION_AT_K', [50, 100, 200, 500])
    for gs_name, gs_adj, gs_eval_set in [
        ("BioGRID",   bg_adj,  bg_eval_set),
        ("STRING",    str_adj, str_eval_set),
        ("GeneMANIA", gm_adj,  gm_eval_set)
    ]:
        for model_key, score_mat in scores_dict.items():
            results[gs_name][model_key]["AUC"] = compute_auc_against_gs(
                score_mat, gs_adj, genes, gs_eval_set
            )
            results[gs_name][model_key]["AUPRC"] = compute_auprc_against_gs(
                score_mat, gs_adj, genes, gs_eval_set
            )
            pak = compute_precision_at_k(score_mat, gs_adj, genes, gs_eval_set, pak_values)
            for k, v in pak.items():
                results[gs_name][model_key][f"P@{k}"] = v

    return results, ta_gat_bin, tagat_score

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

    if CFG.LOCAL_MODE:
        from data_loader import load_local_counts_and_run_deseq2
        sig_degs, expr_groups_dict = load_local_counts_and_run_deseq2()
        dataset_id = CFG.DATASET_NAME
    else:
        sig_degs, expr_groups_dict = load_geo_and_run_deseq2()
        dataset_id = CFG.GEO_ID
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
            
        g_res, g_adj = run_multi_seed_pipeline(
            g_name, dataset_id, g_expr,
            bg_adj, bg_eval_set, string_adj, string_eval_set,
            gm_adj, gm_eval_set, degs_genes, tf_set, device,
            sig_degs=sig_degs
        )
        all_final_results[g_name] = g_res
        adj_dict[g_name] = g_adj

    print_final_summary_tables(all_final_results)
    
    from biological_relevance import analyze_hubs, compute_differential_network
    if adj_dict:
        analyze_hubs(adj_dict, degs_genes, tf_set, config.OUT_DIR)
        # Compute differential network to identify rewired interactions between conditions
        compute_differential_network(adj_dict, degs_genes, config.OUT_DIR)

    print("\n[COMPLETADO] Todas las ejecuciones de TA-GAT han terminado con éxito.")

if __name__ == "__main__":
    main()
