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
