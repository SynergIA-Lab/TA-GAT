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
import requests

import GEOparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

from config import CFG, DATA_DIR

def _download_file(url: str, dest_path: Path):
    """
    Downloads a file from a URL to a local destination path using requests.
    Uses a standard browser User-Agent header to bypass bot blocking.

    Args:
        url (str): The URL of the file to download.
        dest_path (Path): The local destination path where the file will be saved.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

##################################################
# PHASE 1: BIOGRID GOLD STANDARD EXTRACTION     #
##################################################

def download_and_parse_biogrid() -> set[tuple[str, str]]:
    """
    Downloads and parses the BioGRID interaction database.

    Checks if the local BioGRID human-specific file exists. If not, it downloads the 
    zip archive from the configured URL, extracts the tab3 text file, and parses it. 
    It filters specifically for Homo sapiens genetic and physical interactions (TaxID 9606), 
    eliminates self-loops, normalizes gene symbols to uppercase, and removes undirected 
    edge duplicates by sorting pairs alphabetically.

    Returns:
        set[tuple[str, str]]: A set of unique, alphabetically sorted undirected 
            gene interaction pairs (node1, node2) representing the BioGRID gold standard.
    """
    zip_path = DATA_DIR / "biogrid_human.zip"
    txt_path = DATA_DIR / f"{CFG.BIOGRID_FILE_MATCH}-4.4.227.tab3.txt"
    
    if not txt_path.exists():
        print("[BioGRID] Descargando base de datos Gold Standard (~40MB)...")
        _download_file(CFG.BIOGRID_URL, zip_path)
        
        print("[BioGRID] Extrayendo base de datos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if CFG.BIOGRID_FILE_MATCH in file_info.filename:
                    zip_ref.extract(file_info, DATA_DIR)
                    break
    
    print("[BioGRID] Parseando interacciones validadas experimentales...")
    df_bg = pd.read_csv(txt_path, sep="\t", low_memory=False, usecols=[7, 8, 11, 15, 16])
    
    df_bg = df_bg[(df_bg.iloc[:, 3] == 9606) & (df_bg.iloc[:, 4] == 9606)]
    
    edges = df_bg.iloc[:, [0, 1]].dropna().astype(str)
    edges.columns = ["source", "target"]
    edges["source"] = edges["source"].str.upper()
    edges["target"] = edges["target"].str.upper()
    
    edges = edges[edges["source"] != edges["target"]]
    
    edges["node1"] = np.where(edges["source"] < edges["target"], edges["source"], edges["target"])
    edges["node2"] = np.where(edges["source"] < edges["target"], edges["target"], edges["source"])
    unique_edges = set(zip(edges["node1"], edges["node2"]))
    
    print(f"  -> BioGRID Red Cargada: {len(unique_edges)} aristas únicas válidas.")
    return unique_edges

##################################################
# PHASE 2: STRING GOLD STANDARD EXTRACTION       #
##################################################

def download_and_parse_string() -> set[tuple[str, str]]:
    """
    Downloads and parses the STRING protein-protein interaction database.

    Downloads both the protein interaction links and protein information info files 
    for Homo sapiens if not locally present. Parses protein descriptions to build a 
    translation map from ENSP identifiers to HUGO Gene Symbols. Parses the interaction 
    links, filtering pairs by the configured score threshold (e.g., >= 400). Self-loops 
    are removed and pairs are alphabetically sorted to represent undirected edges.

    Returns:
        set[tuple[str, str]]: A set of unique, alphabetically sorted undirected 
            gene interaction pairs (node1, node2) representing the STRING gold standard.
    """
    links_path = DATA_DIR / "9606.protein.links.v12.0.txt.gz"
    info_path  = DATA_DIR / "9606.protein.info.v12.0.txt.gz"
    
    if not links_path.exists():
        print("[STRING] Descargando BD de interacciones (~80MB)...")
        _download_file(CFG.STRING_LINKS_URL, links_path)
    if not info_path.exists():
        print("[STRING] Descargando Info de proteínas...")
        _download_file(CFG.STRING_INFO_URL, info_path)
        
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

##################################################
# PHASE 3: GENEMANIA GOLD STANDARD EXTRACTION   #
##################################################

def download_and_parse_genemania_coexp() -> set[tuple[str, str]]:
    """
    Downloads and parses the GeneMANIA human co-expression databases.

    Downloads the identifier mapping file if not locally present. Parses the mapping 
    to map preferred names to HUGO Gene Symbols. Iterates through the list of 
    configured co-expression network URLs, downloading and reading each. Extracts 
    and translates interaction pairs, removing self-loops and sorting pairs 
    alphabetically to represent undirected edges.

    Returns:
        set[tuple[str, str]]: A set of unique, alphabetically sorted undirected 
            gene interaction pairs (node1, node2) representing the GeneMANIA gold standard.
    """
    mappings_path = DATA_DIR / "genemania_identifier_mappings.txt"
    if not mappings_path.exists():
        print("[GeneMANIA] Descargando ID mappings (~11MB)...")
        _download_file(CFG.GENEMANIA_ID_MAPPING_URL, mappings_path)
    
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
            _download_file(url, local_path)
        
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

##################################################
# PHASE 4: GROUND TRUTH INTERSECTION & MAPPING  #
##################################################

def create_ground_truth_adj(genes: list[str], gs_edges: set[tuple[str, str]]) -> tuple[np.ndarray, set[str]]:
    """
    Maps a global gold standard edge set to the local differentially expressed genes.

    Creates a binary ground truth adjacency matrix representing verified physical or 
    functional interactions restricted to the local gene list. Also extracts the set 
    of 'evaluable genes' (the intersection of the local gene list and the total genes 
    present in the gold standard) for fair validation.

    Args:
        genes (list[str]): List of local gene names.
        gs_edges (set[tuple[str, str]]): Set of unique gold standard interaction pairs.

    Returns:
        tuple[np.ndarray, set[str]]: A tuple containing:
            - adj (np.ndarray): Binary symmetric adjacency matrix of shape (num_genes, num_genes).
            - evaluable_genes_set (set[str]): Sub-set of local genes present in the gold standard.
    """
    gold_genes = set()
    for u, v in gs_edges:
        gold_genes.add(u)
        gold_genes.add(v)
        
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
            
    return adj, evaluable_genes_set

##################################################
# PHASE 5: GEO DATA LOADING & DESEQ2 PROCESSING #
##################################################

def load_geo_and_run_deseq2() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Downloads expression data from GEO and runs PyDESeq2 to identify DEGs.

    Downloads the specified GEO accession ID and parses sample metadata. Splitting 
    GSM samples into cohorts based on the interactively configured variable, it 
    downloads supplementary processed matrices if GSM tables are empty. Autodetects 
    Ensembl IDs and queries the MyGene.info API in chunks to translate them to HUGO 
    Gene Symbols. Rounds expression counts and executes PyDESeq2, contrasting the case 
    cohort against the control cohort. Identifies differentially expressed genes (DEGs) 
    using adjusted p-value and log2 fold-change cutoffs, capping the unified DEG count 
    at 1500 to optimize memory. Returns the DEGs table and cohort-specific expression matrices.

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: A tuple containing:
            - combined_degs (pd.DataFrame): Statistics table for the unified differentially expressed genes.
            - expr_groups_dict (dict[str, pd.DataFrame]): Dictionary mapping cohort names to their 
              respective expression matrices of shape (num_degs, num_cohort_samples).
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
                    for g_name, g_vals in CFG.GEO_GROUPS.items():
                        if val in g_vals or any(val == str(v).strip() for v in g_vals):
                            label = g_name
                            break
        
        if label != "discard":
            meta_rows.append({"gsm": gsm_name, CFG.CONDITION_COL: label})

    if not samples:
        print("\n[WARN] Las tablas GSM están vacías. Buscando matriz de expresión suplementaria en GEO...")
        
        supp_files = []
        for k, v in gse.metadata.items():
            if k.startswith("supplementary_file"):
                supp_files.extend(v)
                
        valid_exts = (".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt")
        supp_url = next((url for url in supp_files if url.lower().endswith(valid_exts)), None)
        
        if supp_url:
            print(f"  -> Descargando {supp_url} ...")
            local_supp = destdir / supp_url.split("/")[-1]
            if not local_supp.exists():
                _download_file(supp_url, local_supp)
                
            sep = "," if ".csv" in str(local_supp).lower() else "\t"
            expr_df = pd.read_csv(local_supp, index_col=0, sep=sep)
            
            if expr_df.shape[1] > 5000 and expr_df.shape[0] < 1000:
                print("  -> Intercambiando la matriz (Muestras x Genes a Genes x Muestras)")
                expr_df = expr_df.T
                
            mapping = {}
            for gsm_name, gsm in gse.gsms.items():
                found = False
                for vals in gsm.metadata.values():
                    if isinstance(vals, list):
                        for val in vals:
                            if val in expr_df.columns:
                                mapping[val] = gsm_name
                                found = True
                                break
                            for col in expr_df.columns:
                                if len(val) > 4 and (col in val or val in col):
                                    mapping[col] = gsm_name
                                    found = True
                                    break
                            if found: break
                    if found: break
            if mapping:
                expr_df.rename(columns=mapping, inplace=True)
                
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
                expr_df = expr_df.groupby(expr_df.index).mean()

    meta_df = pd.DataFrame(meta_rows).set_index("gsm")
    
    common_samples = meta_df.index.intersection(expr_df.columns)
    expr_df = expr_df[common_samples]
    meta_df = meta_df.loc[common_samples]
        
    print(f"  -> Matriz bruta extraída y mapeada: {expr_df.shape}")
    
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
        combined_degs = pd.DataFrame(index=counts_clean.index.tolist()[:1000])
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
            combined_degs = pd.concat(valid_degs)
            combined_degs = combined_degs[~combined_degs.index.duplicated(keep="first")]
        else:
            print("[WARN] No se extrajeron DEGs de los contrastes. Usando los 500 genes mas variables.")
            variances = counts_clean.var(axis=1).sort_values(ascending=False).head(500)
            combined_degs = pd.DataFrame(index=variances.index)
            
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

##################################################
# PHASE 6: TRANSCRIPTION FACTOR LIST UTILITIES   #
##################################################

def load_tf_list(path: str) -> set[str]:
    """
    Loads a reference list of known human transcription factor (TF) names.

    Parses a local text file containing TF gene names (one name per line), 
    converts them to uppercase, and returns them as a set for Prior Boosting.

    Args:
        path (str): File system path to the transcription factor text file.

    Returns:
        set[str]: A set of transcription factor gene names (str).
    """
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

##################################################
# PHASE 7: BASELINE GRAPH & PYG DATA BUILDER     #
##################################################

def build_correlation_baseline(expr: pd.DataFrame, tf_set: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Data]:
    """
    Constructs statistical co-expression baseline networks and prepares GNN features.

    Calculates Pearson and Spearman correlation matrices across genes, applies 
    WGCNA soft-thresholding (power beta=4) to amplify strong links, and promedies 
    them to build an Ensemble correlation matrix. Promotes regulatory priors by 
    multiplying correlation values involving transcription factors (TFs) by 1.5. 
    Binarizes networks using the 95th percentile threshold. Finally, calculates 
    Z-score normalized expression, node degrees, and eigenvector centralities to 
    build and return a PyTorch Geometric Data object.

    Args:
        expr (pd.DataFrame): Expression matrix of shape (num_genes, num_samples).
        tf_set (set[str]): Set of verified transcription factor gene names.

    Returns:
        tuple: A 7-element tuple containing:
            - ensemble_bin (np.ndarray): Binarized ensemble adjacency matrix of shape (num_genes, num_genes).
            - pearson_bin (np.ndarray): Binarized Pearson adjacency matrix.
            - spearman_bin (np.ndarray): Binarized Spearman adjacency matrix.
            - ensemble_corr (np.ndarray): Continuous boosted ensemble correlation matrix.
            - corr_p_boost (np.ndarray): Continuous boosted Pearson correlation matrix.
            - corr_s_boost (np.ndarray): Continuous boosted Spearman correlation matrix.
            - pyg_data (Data): PyTorch Geometric graph data object containing node features, 
              edge indices, and positive edge weights.
    """
    corr_p_vals = np.corrcoef(expr.values)
    corr_p_vals = np.nan_to_num(corr_p_vals, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_p_vals, 0.0)
    corr_p = np.abs(corr_p_vals)
    
    ranks = np.argsort(np.argsort(expr.values, axis=1), axis=1).astype(float)
    corr_s_vals = np.corrcoef(ranks)
    corr_s_vals = np.nan_to_num(corr_s_vals, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_s_vals, 0.0)
    corr_s = np.abs(corr_s_vals)
    
    beta = getattr(CFG, 'WGCNA_POWER_BETA', 4)
    corr_p_powered = corr_p ** beta
    corr_s_powered = corr_s ** beta
    
    ensemble_corr = (corr_p_powered + corr_s_powered) / 2.0
    
    genes = expr.index.tolist()
    is_tf = np.array([g in tf_set for g in genes])
    tf_mask = is_tf[:, None] | is_tf[None, :] 
    
    ensemble_corr[tf_mask] *= CFG.TF_BOOST_FACTOR
    ensemble_corr = np.clip(ensemble_corr, 0.0, 1.0)
    
    corr_p_boost = corr_p.copy()
    corr_p_boost[tf_mask] *= CFG.TF_BOOST_FACTOR
    corr_p_boost = np.clip(corr_p_boost, 0.0, 1.0)
    
    corr_s_boost = corr_s.copy()
    corr_s_boost[tf_mask] *= CFG.TF_BOOST_FACTOR
    corr_s_boost = np.clip(corr_s_boost, 0.0, 1.0)
    
    p_val_e = np.percentile(ensemble_corr.flatten(), CFG.CORR_THRESHOLD_PERCENTILE)
    ensemble_bin = (ensemble_corr >= p_val_e).astype(float)
    
    p_val_p = np.percentile(corr_p_boost.flatten(), CFG.CORR_THRESHOLD_PERCENTILE)
    pearson_bin = (corr_p_boost >= p_val_p).astype(float)
    
    p_val_s = np.percentile(corr_s_boost.flatten(), CFG.CORR_THRESHOLD_PERCENTILE)
    spearman_bin = (corr_s_boost >= p_val_s).astype(float)
    
    edge_indices = np.where(ensemble_bin > 0)
    pos_edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
    pos_edge_attr = torch.tensor(ensemble_corr[edge_indices], dtype=torch.float32).unsqueeze(1)
    
    features = expr.values
    features_mean = features.mean(axis=1, keepdims=True)
    features_std = features.std(axis=1, keepdims=True) + 1e-9
    norm_features = (features - features_mean) / features_std
    
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

##################################################
# PHASE 8: ALGORITHMIC BASELINES (ARACNE, WGCNA)#
##################################################

def build_aracne_baseline(expr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Infers a gene regulatory network using the ARACNE algorithm.

    First calculates Gaussian Mutual Information (MI) from the absolute correlation 
    coefficients as: MI = -0.5 * ln(1 - r^2). Then, applies the Data Processing 
    Inequality (DPI) to prune indirect connections (the weakest link in any triangle 
    is removed if it is weaker than the other two multiplied by a tolerance). 
    Binarizes the resulting MI matrix selecting the top-K highest-scoring edges to 
    match the target density.

    Args:
        expr (pd.DataFrame): Gene expression matrix of shape (num_genes, num_samples).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - aracne_bin (np.ndarray): Binary symmetric adjacency matrix of shape (num_genes, num_genes).
            - out_mi (np.ndarray): Continuous pruned Mutual Information matrix of shape (num_genes, num_genes).
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

def build_wgcna_baseline(expr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Infers a co-expression network using the WGCNA soft-thresholding protocol.

    Calculates the absolute Pearson correlation matrix and raises all coefficients 
    to the soft-thresholding power beta (typically 4). This soft-thresholding 
    reduces noise and strengthens strong correlations. Binarizes the resulting 
    matrix by extracting the top-K highest-scoring edges to match the target density.

    Args:
        expr (pd.DataFrame): Gene expression matrix of shape (num_genes, num_samples).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - wgcna_bin (np.ndarray): Binary symmetric adjacency matrix of shape (num_genes, num_genes).
            - wgcna_adj (np.ndarray): Continuous soft-thresholded correlation matrix.
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
