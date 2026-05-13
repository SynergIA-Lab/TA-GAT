import networkx as nx
import community as community_louvain
import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from pathlib import Path

def run_functional_analysis(adj_matrix: np.ndarray, genes: list, output_dir: Path):
    """
    Realiza un análisis funcional completo de la red:
    1. Clustering (Louvain) para identificar módulos.
    2. Visualización de la red por comunidades.
    3. Análisis de enriquecimiento (Enrichr) para los clusters principales.
    """
    print(f"\n[Análisis Funcional] Procesando red de {len(genes)} genes...")
    
    # 1. Construir grafo de NetworkX
    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: gene for i, gene in enumerate(genes)}
    G = nx.relabel_nodes(G, mapping)

    # 2. Detección de Comunidades (Louvain)
    print("  -> Calculando clusters (Louvain)...")
    partition = community_louvain.best_partition(G)
    
    # Agrupar genes por cluster
    clusters = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(node)
    
    # Guardar asignación de clusters
    cluster_rows = [{"Gene": gene, "Cluster": cid} for cid, g_list in clusters.items() for gene in g_list]
    df_clusters = pd.DataFrame(cluster_rows)
    df_clusters.to_csv(output_dir / "functional_clusters.csv", index=False)
    
    # 3. Visualización de Módulos
    print("  -> Generando visualización de módulos...")
    plt.figure(figsize=(12, 10))
    # Usamos un layout que agrupe visualmente los nodos
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # Colores por cluster
    node_colors = [partition[node] for node in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, cmap=plt.cm.jet)
    nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='gray')
    
    plt.colorbar(nodes, label="Cluster ID")
    plt.title(f"Módulos Funcionales - Louvain (Total clusters: {len(clusters)})", fontsize=15)
    plt.axis('off')
    plt.savefig(output_dir / "tumor_network_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Enriquecimiento por Cluster (Enrichr)
    main_clusters = {k: v for k, v in clusters.items() if len(v) >= 15}
    print(f"  -> {len(main_clusters)} clusters principales identificados (>=15 genes).")
    
    enrich_dir = output_dir / "enrichment"
    enrich_dir.mkdir(parents=True, exist_ok=True)
    
    gene_sets = ['GO_Biological_Process_2023', 'KEGG_2021_Human']
    
    for cid, gene_list in main_clusters.items():
        print(f"    - Análisis para Cluster {cid} ({len(gene_list)} genes)...")
        
        # Ejecutamos enriquecimiento para cada base de datos por separado para mayor claridad
        for gset in gene_sets:
            db_short = "GO" if "GO" in gset else "KEGG"
            try:
                enr = gp.enrichr(gene_list=gene_list,
                                 gene_sets=gset,
                                 organism='human',
                                 outdir=str(enrich_dir / f"cluster_{cid}_{db_short}"),
                                 cutoff=0.05)
                
                res = enr.results
                # Filtrar significativos para el plot
                sig_res = res[res['Adjusted P-value'] < 0.05]
                
                if not sig_res.empty:
                    from gseapy import dotplot
                    dotplot(sig_res, 
                            title=f"Enriquecimiento {db_short} - Cluster {cid}",
                            column="Adjusted P-value",
                            size=10,
                            top_term=15,
                            figsize=(8, 10),
                            ofname=str(enrich_dir / f"dotplot_cluster_{cid}_{db_short}.png"))
                    print(f"      [OK] {db_short}: {len(sig_res)} términos significativos encontrados.")
                else:
                    print(f"      [INFO] {db_short}: Sin términos significativos (p-adj < 0.05).")
                    
            except Exception as e:
                # Capturamos el Warning de gseapy que a veces viene como excepción en entornos interactivos
                if "No enrich terms" in str(e):
                    print(f"      [INFO] {db_short}: Sin enriquecimiento para este cluster.")
                else:
                    print(f"      [ERR] {db_short}: {e}")

    print(f"  -> Resultados de análisis funcional guardados en: {output_dir}")
    return df_clusters
