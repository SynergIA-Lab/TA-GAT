import os
import re
import zipfile
import gzip
import urllib.request
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx

import GEOparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

from config import CFG, DATA_DIR

# =============================================================================
# BioGRID Gold Standard Loader
# =============================================================================

def download_and_parse_biogrid() -> pd.DataFrame:
    """
    Descarga la última release de BioGRID (Homo sapiens), localiza las
    interacciones genéticas y físicas, y retorna un DataFrame estandarizado
    de aristas reales.
    """
    zip_path = DATA_DIR / "biogrid_human.zip"
    txt_path = DATA_DIR / f"{CFG.BIOGRID_FILE_MATCH}-4.4.227.tab3.txt"
    
    if not txt_path.exists():
        print("[BioGRID] Descargando base de datos Gold Standard (~40MB)...")
        urllib.request.urlretrieve(CFG.BIOGRID_URL, zip_path)
        
        print("[BioGRID] Extrayendo base de datos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraer solo el archivo de Homo Sapiens
            for file_info in zip_ref.infolist():
                if CFG.BIOGRID_FILE_MATCH in file_info.filename:
                    zip_ref.extract(file_info, DATA_DIR)
                    break
    
    print("[BioGRID] Parseando interacciones validadas experimentales...")
    # Usamos low_memory=False y leemos columnas esenciales de BioGRID:
    # Col 7 y 8: Official Symbol Interactor A & B
    # Col 11: Experimental System Type (Física o Genética)
    # Col 15: Organism ID Interactor A (9606 es humano)
    df_bg = pd.read_csv(txt_path, sep="\t", low_memory=False, usecols=[7, 8, 11, 15, 16])
    
    # Filtrar solo interacciones puramente humanas
    df_bg = df_bg[(df_bg.iloc[:, 3] == 9606) & (df_bg.iloc[:, 4] == 9606)]
    
    # Limpiamos y estandarizamos la red
    edges = df_bg.iloc[:, [0, 1]].dropna().astype(str)
    edges.columns = ["source", "target"]
    edges["source"] = edges["source"].str.upper()
    edges["target"] = edges["target"].str.upper()
    
    # Quitar auto-bucles y duplicados para crear una adyacencia binaria simétrica limpia
    edges = edges[edges["source"] != edges["target"]]
    
    # Ordenar A-B para eliminar (B, A) como red no dirigida
    edges["node1"] = np.where(edges["source"] < edges["target"], edges["source"], edges["target"])
    edges["node2"] = np.where(edges["source"] < edges["target"], edges["target"], edges["source"])
    unique_edges = set(zip(edges["node1"], edges["node2"]))
    
    print(f"  -> BioGRID Red Cargada: {len(unique_edges)} aristas únicas válidas.")
    return unique_edges


def download_and_parse_string() -> set:
    """
    Descarga STRING y mapea las proteínas a símbolos de genes (Hugo).
    Filtra interacciones por score (medio/alto según config).
    """
    links_path = DATA_DIR / "9606.protein.links.v12.0.txt.gz"
    info_path  = DATA_DIR / "9606.protein.info.v12.0.txt.gz"
    
    if not links_path.exists():
        print("[STRING] Descargando BD de interacciones (~80MB)...")
        urllib.request.urlretrieve(CFG.STRING_LINKS_URL, links_path)
    if not info_path.exists():
        print("[STRING] Descargando Info de proteínas...")
        urllib.request.urlretrieve(CFG.STRING_INFO_URL, info_path)
        
    print("[STRING] Parseando alias de proteínas a genes...")
    protein_to_gene = {}
    with gzip.open(info_path, 'rt') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                protein_to_gene[parts[0]] = parts[1].upper()
                
    print(f"[STRING] Procesando interacciones experimentales (Score >= {CFG.STRING_SCORE_THRESH})...")
    unique_edges = set()
    with gzip.open(links_path, 'rt') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                p1, p2, score_str = parts
                score = int(score_str)
                if score >= CFG.STRING_SCORE_THRESH:
                    g1 = protein_to_gene.get(p1)
                    g2 = protein_to_gene.get(p2)
                    if g1 and g2 and g1 != g2:
                        if g1 < g2:
                            unique_edges.add((g1, g2))
                        else:
                            unique_edges.add((g2, g1))
                            
    print(f"  -> STRING Red Cargada: {len(unique_edges)} aristas únicas válidas.")
    return unique_edges

def create_ground_truth_adj(genes: list, gs_edges: set) -> tuple[np.ndarray, set]:
    """
    Mapea un set de interacciones (BioGRID o STRING) a nuestra lista de genes DEGs.
    Retorna:
     1. La matriz binaria (G x G) de ground-truth.
     2. El set de "Genes Evaluables" para FAIR EVALUATION.
    """
    
    # ¿Qué genes realmente existen en este Gold Standard?
    gold_genes = set()
    for u, v in gs_edges:
        gold_genes.add(u)
        gold_genes.add(v)
        
    # Intersecar con nuestros DEGs
    evaluable_genes_set = gold_genes.intersection(set(genes))
    
    gene_map_idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)
    adj = np.zeros((N, N), dtype=np.float32)
    
    edges_found = 0
    for u, v in gs_edges:
        if u in gene_map_idx and v in gene_map_idx:
            adj[gene_map_idx[u], gene_map_idx[v]] = 1.0
            adj[gene_map_idx[v], gene_map_idx[u]] = 1.0
            edges_found += 1
            
    # print(f"  -> Mapeo a DEGs locales: {edges_found} aristas GS halladas. ({len(evaluable_genes_set)} evaluables)")
    
    return adj, evaluable_genes_set



# =============================================================================
# GEO & PyDESeq2 Loaders
# =============================================================================

def load_geo_and_run_deseq2() -> tuple[pd.DataFrame, dict]:
    """
    Descarga GEO, mapea a metadata estricta por grupos configurados interactivamente.
    Aplica PyDESeq2 y retorna:
    - de_results: Tabla de resultados de DESeq2 (union de todos los DEGs)
    - expr_dict: Diccionario de {nombre_grupo: DataFrame expresion (genes x muestras)}
    """
    geo_id = CFG.GEO_ID
    destdir = DATA_DIR
    
    print(f"\n[GEO] Validando dataset {geo_id}...")
    os.makedirs(destdir, exist_ok=True)
    gse = GEOparse.get_GEO(geo=geo_id, destdir=str(destdir), silent=True)
    
    samples, names, meta_rows = [], [], []
    for gsm_name, gsm in gse.gsms.items():
        if not gsm.table.empty and 'VALUE' in gsm.table.columns:
            serie = pd.to_numeric(
                gsm.table.set_index('ID_REF')['VALUE'], errors='coerce'
            )
            samples.append(serie)
            names.append(gsm_name)
            
        chars = gsm.metadata.get("characteristics_ch1", [])
        label = "discard"
        for c in chars:
            if ":" in c:
                key, val = c.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key == CFG.GEO_METADATA_KEY:
                    # Buscar en CFG.GEO_GROUPS
                    for g_name, g_vals in CFG.GEO_GROUPS.items():
                        if val in g_vals or any(val == str(v).strip() for v in g_vals):
                            label = g_name
                            break
        
        if label != "discard":
            meta_rows.append({"gsm": gsm_name, CFG.CONDITION_COL: label})

    if not samples:
        print("\n[WARN] Las tablas GSM están vacías. Buscando matriz de expresión suplementaria en GEO...")
        
        # Buscar archivo(s) suplementario(s)
        supp_files = []
        for k, v in gse.metadata.items():
            if k.startswith("supplementary_file"):
                supp_files.extend(v)
                
        # Priorizar procesados
        valid_exts = (".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt")
        supp_url = next((url for url in supp_files if url.lower().endswith(valid_exts)), None)
        
        if supp_url:
            print(f"  -> Descargando {supp_url} ...")
            local_supp = destdir / supp_url.split("/")[-1]
            if not local_supp.exists():
                urllib.request.urlretrieve(supp_url, str(local_supp))
                
            sep = "," if ".csv" in str(local_supp).lower() else "\t"
            expr_df = pd.read_csv(local_supp, sep=sep, index_col=0)
            
            # Autodetectar orientación (genes son +5k), transponer si genes están en columnas
            if expr_df.shape[1] > 5000 and expr_df.shape[0] < 1000:
                print("  -> Intercambiando la matriz (Muestras x Genes a Genes x Muestras)")
                expr_df = expr_df.T
                
            # Mapear las columnas de expr_df a los IDs de muestras GSM
            mapping = {}
            for gsm_name, gsm in gse.gsms.items():
                found = False
                for vals in gsm.metadata.values():
                    if isinstance(vals, list):
                        for val in vals:
                            # Match directo completo
                            if val in expr_df.columns:
                                mapping[val] = gsm_name
                                found = True
                                break
                            # Match parcial (substring cruce) si no hay directo
                            for col in expr_df.columns:
                                if len(val) > 4 and (col in val or val in col):
                                    mapping[col] = gsm_name
                                    found = True
                                    break
                            if found: break
                    if found: break
            if mapping:
                expr_df.rename(columns=mapping, inplace=True)
                
            # Mapeo de Ensembl a Gene Symbol (Hugo)
            if any(str(idx).startswith("ENSG") for idx in expr_df.index[:20]):
                print("  -> Autodetectados Ensembl IDs. Traduciendo a Gene Symbols mediante MyGene API...")
                
                ensg_ids = list(expr_df.index)
                mapping_ensg_to_symbol = {}
                chunk_size = 1000
                for i in range(0, len(ensg_ids), chunk_size):
                    chunk = ensg_ids[i:i + chunk_size]
                    try:
                        data = json.dumps({"q": chunk, "scopes": "ensembl.gene", "fields": "symbol"}).encode('utf-8')
                        req = urllib.request.Request("https://mygene.info/v3/query", data=data, headers={'Content-Type': 'application/json'})
                        response = urllib.request.urlopen(req)
                        res = json.loads(response.read().decode('utf-8'))
                        for item in res:
                            if 'symbol' in item and 'query' in item:
                                mapping_ensg_to_symbol[item['query']] = item['symbol']
                    except Exception as e:
                        print(f"[WARN] Fallo mapeando lote de genes vía MyGene API: {e}")
                        
                expr_df.index = expr_df.index.map(lambda x: mapping_ensg_to_symbol.get(x, x))
                expr_df = expr_df.groupby(expr_df.index).mean()
        else:
            raise ValueError("[ERROR] No se han encontrado archivos suplementarios para descargar la matriz de expresión.")
            
    else:
        expr_df = pd.concat(samples, axis=1)
        expr_df.columns = names
        expr_df.dropna(inplace=True)
        
        # Mapear Probe IDs
        if gse.gpls:
            gpl_name = list(gse.gpls.keys())[0]
            gpl = gse.gpls[gpl_name]
            if gpl.table is not None and not gpl.table.empty and 'Gene Symbol' in gpl.table.columns:
                gpl_map = gpl.table[['ID', 'Gene Symbol']].dropna()
                gpl_map = gpl_map[~gpl_map['Gene Symbol'].str.contains('///')]
                gpl_map = gpl_map[gpl_map['Gene Symbol'] != '']
                gpl_dict = dict(zip(gpl_map['ID'], gpl_map['Gene Symbol']))
    
                expr_df.index = expr_df.index.map(gpl_dict)
                expr_df.dropna(inplace=True)
                expr_df = expr_df.groupby(expr_df.index).mean() # Promedio duplicados

    meta_df = pd.DataFrame(meta_rows).set_index("gsm")
    
    # Asegurarnos de usar solo las muestras que existen en expr_df
    common_samples = meta_df.index.intersection(expr_df.columns)
    expr_df = expr_df[common_samples]
    meta_df = meta_df.loc[common_samples]
        
    print(f"  -> Matriz bruta extraída y mapeada: {expr_df.shape}")
    
    # ── PyDESeq2 ─────────────────────────────────────────────────────────────
    print("\n[PyDESeq2] Ejecutando Differential Expression Analysis...")
    counts_clean = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    if (counts_clean < 0).any().any():
        print("  -> Autodetectados valores negativos (log-transform). Revirtiendo a cuentas raw approximate (2^x)...")
        counts_clean = 2 ** counts_clean
        
    counts_clean = np.maximum(0, counts_clean).round().astype(int)
    counts_clean = counts_clean.loc[counts_clean.sum(axis=1) >= CFG.MIN_TOTAL_COUNTS_PER_GENE, :]
    
    counts_sxg = counts_clean.T
    inference = DefaultInference(n_cpus=CFG.N_CPUS)
    dds = DeseqDataSet(
        counts=counts_sxg,
        metadata=meta_df,
        design=f"~ {CFG.CONDITION_COL}",
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()
    
    control_group = CFG.GEO_CONTROL_GROUP
    groups_to_compare = [g for g in CFG.GEO_GROUPS.keys() if g != control_group]
    
    all_sig_degs = []
    
    if len(groups_to_compare) == 0:
        print("[WARN] Solo se ha definido un grupo o no hay grupos contra los que comparar.")
        combined_degs = pd.DataFrame(index=counts_clean.index.tolist()[:1000]) # dummy
    else:    
        for case_group in groups_to_compare:
            print(f"  -> Contrastando {case_group} vs {control_group}...")
            try:
                stat_res = DeseqStats(
                    dds,
                    contrast=(CFG.CONDITION_COL, case_group, control_group),
                    inference=inference,
                    alpha=CFG.ALPHA,
                )
                stat_res.summary()
                results_df = stat_res.results_df.copy()
                
                sig_degs = results_df[results_df["padj"].notna() & (results_df["padj"] <= CFG.ALPHA)]
                sig_degs = sig_degs[sig_degs["log2FoldChange"].abs() >= CFG.LOG2FC_THRESH]
                all_sig_degs.append(sig_degs)
            except Exception as e:
                print(f"[ERROR] Falló contraste {case_group} vs {control_group}: {e}")

        valid_degs = [df for df in all_sig_degs if len(df) > 0]
        if valid_degs:
            # Union de todos los DEGs
            combined_degs = pd.concat(valid_degs)
            combined_degs = combined_degs[~combined_degs.index.duplicated(keep="first")]
        else:
            # Si falla todo o no hay genes sig, coger top varianza o un set dummy
            print("[WARN] No se extrajeron DEGs de los contrastes. Usando los 500 genes mas variables.")
            variances = counts_clean.var(axis=1).sort_values(ascending=False).head(500)
            combined_degs = pd.DataFrame(index=variances.index)
            
    # Capping DEGs size to avoid OOM
    if len(combined_degs) > 1500:
        if "padj" in combined_degs.columns:
            combined_degs = combined_degs.sort_values("padj").head(1500)
        else:
            combined_degs = combined_degs.head(1500)
    elif len(combined_degs) < 50:
        print("[WARN] Muy pocos DEGs. Capping.")
        
    deg_names = combined_degs.index.tolist()
    
    expr_groups_dict = {}
    for g_name in CFG.GEO_GROUPS.keys():
        g_samples = meta_df[meta_df[CFG.CONDITION_COL] == g_name].index
        expr_groups_dict[g_name] = expr_df.loc[deg_names, g_samples]
        
    print(f"  -> DEGs finales unificados procesados: {len(deg_names)}")
    for g_name, mat in expr_groups_dict.items():
        print(f"  -> Matriz '{g_name}': {mat.shape}")
        
    return combined_degs, expr_groups_dict

# =============================================================================
# Graph Format Preparation & TFs
# =============================================================================

def load_tf_list(path: str) -> set:
    """Carga la lista de Factores de Transcripción conocidos."""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return set([line.strip().upper() for line in f if line.strip()])
        else:
            print(f"[WARN] No se encontró la lista de TFs en {path}. Asumiendo sin TFs.")
            return set()
    except Exception as e:
        print(f"[WARN] Error leyendo TFs: {e}")
        return set()

def build_correlation_baseline(expr: pd.DataFrame, tf_set: set) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Data]:
    """
    Construye las redes base (Pearson, Spearman y Ensemble).
    Aplica Boosts de TFs.
    Extrae un grafo de PyTorch Geometric apoyado sobre el Ensemble.
    Retorna: (ensemble_bin, pearson_bin, spearman_bin, ensemble_corr_raw, pyg_data)
    """
    # Pearson
    corr_p_vals = np.corrcoef(expr.values)
    corr_p_vals = np.nan_to_num(corr_p_vals, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_p_vals, 0.0)
    corr_p = np.abs(corr_p_vals)
    
    # Spearman
    ranks = np.argsort(np.argsort(expr.values, axis=1), axis=1).astype(float)
    corr_s_vals = np.corrcoef(ranks)
    corr_s_vals = np.nan_to_num(corr_s_vals, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_s_vals, 0.0)
    corr_s = np.abs(corr_s_vals)
    
    # WGCNA Soft-thresholding
    beta = getattr(CFG, 'WGCNA_POWER_BETA', 4)
    corr_p_powered = corr_p ** beta
    corr_s_powered = corr_s ** beta
    
    # Ensemble (Baseline clásico adaptado con Soft-Thresholding)
    ensemble_corr = (corr_p_powered + corr_s_powered) / 2.0
    
    # TF Boosting individual por matrices por si lo usamos después
    genes = expr.index.tolist()
    is_tf = np.array([g in tf_set for g in genes])
    
    # Si al menos un nodo en el par (i, j) es TF, aplicamos el boost
    tf_mask = is_tf[:, None] | is_tf[None, :] 
    
    # Aplicar TF boost a Ensemble
    ensemble_corr[tf_mask] *= CFG.TF_BOOST_FACTOR
    ensemble_corr = np.clip(ensemble_corr, 0.0, 1.0)
    
    # Aplicar a las correlaciones independientes
    corr_p_boost = corr_p.copy()
    corr_p_boost[tf_mask] *= CFG.TF_BOOST_FACTOR
    corr_p_boost = np.clip(corr_p_boost, 0.0, 1.0)
    
    corr_s_boost = corr_s.copy()
    corr_s_boost[tf_mask] *= CFG.TF_BOOST_FACTOR
    corr_s_boost = np.clip(corr_s_boost, 0.0, 1.0)
    
    # Umbralizado al exacto mismo percentil para que TODAS las redes tengan la misma Sparsity inicial
    p_val_e = np.percentile(ensemble_corr.flatten(), CFG.CORR_THRESHOLD_PERCENTILE)
    ensemble_bin = (ensemble_corr >= p_val_e).astype(float)
    
    p_val_p = np.percentile(corr_p_boost.flatten(), CFG.CORR_THRESHOLD_PERCENTILE)
    pearson_bin = (corr_p_boost >= p_val_p).astype(float)
    
    p_val_s = np.percentile(corr_s_boost.flatten(), CFG.CORR_THRESHOLD_PERCENTILE)
    spearman_bin = (corr_s_boost >= p_val_s).astype(float)
    
    # PyG Graph logic basado en el ENSEMBLE
    edge_indices = np.where(ensemble_bin > 0)
    pos_edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
    pos_edge_attr = torch.tensor(ensemble_corr[edge_indices], dtype=torch.float32).unsqueeze(1)
    
    # Feature Normalization (Z-score por gen) para evitar colapsos
    features = expr.values
    features_mean = features.mean(axis=1, keepdims=True)
    features_std = features.std(axis=1, keepdims=True) + 1e-9
    norm_features = (features - features_mean) / features_std
    
    # Extra Topological Features (Normalized Degree + Eigenvector Centrality)
    G_base = nx.from_numpy_array(ensemble_bin)
    degrees = np.array([d for n, d in G_base.degree()])
    if degrees.max() > 0:
        degrees = degrees / degrees.max()
        
    try:
        dict_eig = nx.eigenvector_centrality(G_base, max_iter=500)
        eig_cent = np.array([dict_eig[i] for i in range(len(G_base))])
    except nx.PowerIterationFailedConvergence:
        eig_cent = np.zeros(len(G_base))
        
    extra_features = np.column_stack((degrees, eig_cent))
    final_features = np.column_stack((norm_features, extra_features))
    
    node_features = torch.tensor(final_features, dtype=torch.float32)
    pyg_data = Data(x=node_features, edge_index=pos_edge_index, edge_attr=pos_edge_attr)
    
    return ensemble_bin, pearson_bin, spearman_bin, ensemble_corr, corr_p_boost, corr_s_boost, pyg_data

def download_and_parse_genemania_coexp() -> set:
    """
    Descarga Mapeos de IDs y redes de Co-expresión de GeneMANIA.
    Mapea de Preferred_Name a Hugo Gene Symbol y retorna el GS de Co-expresión.
    """
    mappings_path = DATA_DIR / "genemania_identifier_mappings.txt"
    if not mappings_path.exists():
        print("[GeneMANIA] Descargando ID mappings (~11MB)...")
        urllib.request.urlretrieve(CFG.GENEMANIA_ID_MAPPING_URL, mappings_path)
    
    print("[GeneMANIA] Parseando ID Mappings (Preferred_Name to Gene Name)...")
    pm_to_gene = {}
    with open(mappings_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pname, name, source = parts[0], parts[1], parts[2]
                if source == "Gene Name":
                    pm_to_gene[pname] = name.upper()
    
    unique_edges = set()
    print("[GeneMANIA] Procesando redes de Co-expresión superiores...")
    for url in CFG.GENEMANIA_COEXP_URLS:
        filename = url.split("/")[-1]
        local_path = DATA_DIR / filename
        if not local_path.exists():
            print(f"  -> Descargando {filename} ...")
            urllib.request.urlretrieve(url, local_path)
        
        with open(local_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    g1 = pm_to_gene.get(parts[0])
                    g2 = pm_to_gene.get(parts[1])
                    if g1 and g2 and g1 != g2:
                        if g1 < g2:
                            unique_edges.add((g1, g2))
                        else:
                            unique_edges.add((g2, g1))
                            
    print(f"  -> GeneMANIA Co-expression Red Cargada: {len(unique_edges)} aristas únicas válidas.")
    return unique_edges

def build_aracne_baseline(expr: pd.DataFrame) -> np.ndarray:
    """
    Implementación de ARACNE (Mutual Information + DPI).
    1. MI Gaussiana: MI = -0.5 * ln(1 - rho^2).
    2. DPI (Data Processing Inequality) para eliminar arcos indirectos.
    """
    corr_matrix = np.abs(np.nan_to_num(np.corrcoef(expr.values), nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr_matrix, 0.0)
    
    corr_matrix = np.clip(corr_matrix, 0.0, 0.9999)
    mi_matrix = -0.5 * np.log(1 - corr_matrix**2)
    
    N = mi_matrix.shape[0]
    out_mi = mi_matrix.copy()
    min_mi = np.percentile(mi_matrix, 75)
    tolerance = 1.0 - getattr(CFG, 'ARACNE_DPI_TOLERANCE', 0.1)
    
    for i in range(N):
        neighbors_i = np.where(out_mi[i] > min_mi)[0]
        for idx_j, j in enumerate(neighbors_i):
            for k in neighbors_i[idx_j+1:]:
                val_ij, val_ik, val_jk = out_mi[i, j], out_mi[i, k], out_mi[j, k]
                if val_ij > 0 and val_ik > 0 and val_jk > 0:
                    m = min(val_ij, val_ik, val_jk)
                    if m == val_ij and val_ij < min(val_ik, val_jk) * tolerance:
                        out_mi[i, j] = out_mi[j, i] = 0
                    elif m == val_ik and val_ik < min(val_ij, val_jk) * tolerance:
                        out_mi[i, k] = out_mi[k, i] = 0
                    elif m == val_jk and val_jk < min(val_ij, val_ik) * tolerance:
                        out_mi[j, k] = out_mi[k, j] = 0
                        
    target_density = getattr(CFG, 'INFERENCE_TARGET_DENSITY', 0.05)
    target_edges = int(N * (N - 1) / 2 * target_density)
    
    upper_mi = out_mi[np.triu_indices(N, k=1)]
    thresh = np.partition(upper_mi, len(upper_mi) - target_edges)[len(upper_mi) - target_edges] if target_edges < len(upper_mi) else 0.0
    
    aracne_bin = (out_mi >= thresh).astype(float)
    np.fill_diagonal(aracne_bin, 0.0)
    return aracne_bin, out_mi

def build_wgcna_baseline(expr: pd.DataFrame) -> np.ndarray:
    """
    Inferencia WGCNA (Soft-Thresholding).
    Aplica una potencia (beta) a la correlación de Pearson.
    """
    corr_p = np.abs(np.nan_to_num(np.corrcoef(expr.values), nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr_p, 0.0)
    
    beta = getattr(CFG, 'WGCNA_POWER_BETA', 4)
    wgcna_adj = corr_p ** beta
    
    N = wgcna_adj.shape[0]
    target_density = getattr(CFG, 'INFERENCE_TARGET_DENSITY', 0.05)
    target_edges = int(N * (N - 1) / 2 * target_density)
    
    upper_vals = wgcna_adj[np.triu_indices(N, k=1)]
    thresh = np.partition(upper_vals, len(upper_vals) - target_edges)[len(upper_vals) - target_edges] if target_edges < len(upper_vals) else 0.0
        
    wgcna_bin = (wgcna_adj >= thresh).astype(float)
    np.fill_diagonal(wgcna_bin, 0.0)
    return wgcna_bin, wgcna_adj
