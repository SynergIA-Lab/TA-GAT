import pandas as pd
import numpy as np
from pathlib import Path

##################################################
# PHASE 1: GENE REGULATORY NETWORK HUBS & TFS   #
##################################################

def analyze_hubs(adj_dict: dict, genes: list, tf_set: set, output_dir: Path) -> pd.DataFrame:
    """
    Identifies and evaluates network hub genes for biological relevance.

    For each experimental condition, this function calculates the node degree of 
    every gene in the final inferred network, extracts the top 20 most connected 
    genes (hubs), determines which of these hubs are known transcription factors (TFs), 
    and exports both condition-specific hub details and a comparative summary.

    Args:
        adj_dict (dict): Dictionary mapping condition names (str) to binary adjacency 
            matrices (np.ndarray) of shape (num_genes, num_genes).
        genes (list): List of gene names (str) corresponding to the indices of the 
            adjacency matrices.
        tf_set (set): Set of transcription factor gene names (str) used for biological 
            prior validation.
        output_dir (Path): Directory path where the CSV reports will be saved.

    Returns:
        pd.DataFrame: A summary DataFrame containing the condition, total hubs, number 
            of TFs, and the percentage of TFs among the hubs.
    """
    summary_rows = []
    
    for condition, adj in adj_dict.items():
        degrees = np.sum(adj, axis=1)
        
        df_genes = pd.DataFrame({
            'Gene': genes,
            'Degree': degrees,
            'is_TF': [g in tf_set for g in genes]
        })
        
        top_20 = df_genes.sort_values(by='Degree', ascending=False).head(20)
        
        top_20.to_csv(output_dir / f"top_20_hubs_{condition.lower()}.csv", index=False)
        
        num_tfs = top_20['is_TF'].sum()
        pct_tfs = (num_tfs / 20) * 100
        
        summary_rows.append({
            'Condición': condition,
            'Total_Hubs': 20,
            'Num_TFs': int(num_tfs),
            'PCT_TFs': f"{pct_tfs:.2f}%"
        })
        
        print(f"\n[Análisis de Hubs - {condition}]")
        print(f"  Top 20 Hubs identificados. TFs encontrados: {num_tfs} ({pct_tfs:.2f}%)")
        print(top_20[['Gene', 'Degree', 'is_TF']].to_string(index=False))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "hubs_comparison.csv", index=False)
    print(f"\n[ÉXITO] Archivo 'hubs_comparison.csv' generado en {output_dir}")
    
    return df_summary


##################################################
# PHASE 2: DIFFERENTIAL NETWORK ANALYSIS        #
##################################################

def compute_differential_network(
    adj_dict: dict,
    genes: list,
    output_dir: Path
) -> pd.DataFrame | None:
    """
    Computes the differential (rewired) gene interaction network between two conditions.

    For each pair of conditions (e.g., Tumor vs Normal), identifies:
    - Edges gained exclusively in condition A (not present in B).
    - Edges lost in condition A (present in B but not A).
    - Net differential degree per gene: number of connections gained minus lost.

    Genes with high absolute net differential degree are the most rewired regulators
    in the biological transition, making them prime experimental candidates.

    Args:
        adj_dict (dict): Mapping condition name → binary adjacency matrix (num_genes, num_genes).
        genes (list): Gene names corresponding to matrix indices.
        output_dir (Path): Directory where CSV reports and differential degree rankings are saved.

    Returns:
        pd.DataFrame | None: DataFrame with one row per gene sorted by |net differential|,
            or None if fewer than two conditions are available.
    """
    if len(adj_dict) < 2:
        print("[Red Diferencial] Se necesitan al menos 2 condiciones. Saltando análisis.")
        return None

    cond_names = list(adj_dict.keys())
    cond_a, cond_b = cond_names[0], cond_names[1]
    adj_a = adj_dict[cond_a].astype(float)
    adj_b = adj_dict[cond_b].astype(float)

    # Edges present in A but not B (gained), and in B but not A (lost)
    gained_in_a = np.clip(adj_a - adj_b, 0, 1)  # exclusive to condition A
    lost_in_a   = np.clip(adj_b - adj_a, 0, 1)  # exclusive to condition B

    gained_degree = gained_in_a.sum(axis=1)
    lost_degree   = lost_in_a.sum(axis=1)
    net_diff      = gained_degree - lost_degree  # + → more connections in A; − → fewer

    df = pd.DataFrame({
        "Gene":            genes,
        f"Degree_{cond_a}": adj_a.sum(axis=1).astype(int),
        f"Degree_{cond_b}": adj_b.sum(axis=1).astype(int),
        "Gained_in_A":     gained_degree.astype(int),
        "Lost_in_A":       lost_degree.astype(int),
        "Net_Differential": net_diff.astype(int)
    }).sort_values("Net_Differential", key=abs, ascending=False)

    out_file = output_dir / f"differential_network_{cond_a}_vs_{cond_b}.csv"
    df.to_csv(out_file, index=False)

    print(f"\n[Red Diferencial] {cond_a} vs {cond_b}")
    print(f"  → Aristas exclusivas de {cond_a}: {int(gained_in_a.sum() / 2)}")
    print(f"  → Aristas exclusivas de {cond_b}: {int(lost_in_a.sum() / 2)}")
    print(f"  → Top 20 genes con mayor rewiring diferencial:")
    print(df.head(20)[["Gene", f"Degree_{cond_a}", f"Degree_{cond_b}", "Net_Differential"]].to_string(index=False))
    print(f"  → Guardado en: {out_file}")

    return df
