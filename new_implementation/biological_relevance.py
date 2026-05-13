import pandas as pd
import numpy as np
from pathlib import Path

def analyze_hubs(adj_dict: dict, genes: list, tf_set: set, output_dir: Path):
    """
    Extrae los Top 20 Hubs por condición y calcula el % de TFs.
    adj_dict: { 'Tumor': adj_matrix, 'Normal': adj_matrix }
    """
    summary_rows = []
    
    for condition, adj in adj_dict.items():
        # Calcular grados (suma de filas ya que es simétrica)
        degrees = np.sum(adj, axis=1)
        
        # Crear DataFrame de genes y grados
        df_genes = pd.DataFrame({
            'Gene': genes,
            'Degree': degrees,
            'is_TF': [g in tf_set for g in genes]
        })
        
        # Sort y extraer Top 20
        top_20 = df_genes.sort_values(by='Degree', ascending=False).head(20)
        
        # Guardar Top 20 detalle para esta condición
        top_20.to_csv(output_dir / f"top_20_hubs_{condition.lower()}.csv", index=False)
        
        # Calcular métricas agregadas
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

    # Generar el archivo final solicitado hubs_comparison.csv
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "hubs_comparison.csv", index=False)
    print(f"\n[ÉXITO] Archivo 'hubs_comparison.csv' generado en {output_dir}")
    
    return df_summary
