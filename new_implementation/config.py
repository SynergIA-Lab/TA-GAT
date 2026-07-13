import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_IMPL_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

class Config:
    """
    Configuration parameters and hyperparameters for the TA-GAT pipeline.

    This class stores paths, experimental group configurations, statistical 
    thresholds, URL endpoints, neural network configurations, and weights 
    for the multi-objective loss function.

    Attributes:
        FAST_MODE (bool): If True, overrides N_SEEDS to [123] (single seed). Use during
            development to verify the pipeline in ~20 min instead of hours. Set to False
            for the final paper run (uses the full N_SEEDS list with 5+ seeds).
        SEED (int): Random seed for software-wide deterministic reproducibility.
        GEO_ID (str): Accession ID of the Gene Expression Omnibus dataset (e.g., 'GSE11121').
        GEO_METADATA_KEY (str): Metadata column selected from characteristics_ch1 for sample split.
        GEO_GROUPS (dict): Mapping of cohort label (e.g., 'Tumor', 'Normal') to selected metadata values.
        GEO_CONTROL_GROUP (str): Reference group used as the baseline control for PyDESeq2 (typically 'Normal').
        CONDITION_COL (str): Column name representing the experimental design variable.
        MIN_TOTAL_COUNTS_PER_GENE (int): Minimum count sum threshold for filtering low-expression genes.
        ALPHA (float): Significance level threshold for adjusted p-value (FDR) in PyDESeq2.
        LOG2FC_THRESH (float): Absolute log2 fold-change cutoff for identifying differentially expressed genes.
        N_CPUS (int): Number of CPU cores used for parallel processing during PyDESeq2 execution.
        BIOGRID_URL (str): Download link for the BioGRID interaction database archive.
        BIOGRID_FILE_MATCH (str): Filename matching pattern to extract the human-specific BioGRID tab3 file.
        STRING_LINKS_URL (str): Download link for the STRING protein-protein interaction database.
        STRING_INFO_URL (str): Download link for the STRING protein information metadata mapping.
        STRING_SCORE_THRESH (int): Cutoff score for filtering high-confidence STRING interactions (scale 0-1000).
        GENEMANIA_ID_MAPPING_URL (str): GeneMANIA identifier mapping file download endpoint.
        GENEMANIA_COEXP_URLS (list): List of download links for individual GeneMANIA human co-expression networks.
        TF_LIST_PATH (str): Local path to the transcription factor name reference list.
        TF_BOOST_FACTOR (float): Amplification factor applied to co-expression weights involving at least one TF.
        CORR_THRESHOLD_PERCENTILE (float): Percentile threshold for binarizing WGCNA-style baseline networks (e.g., 95).
        INPUT_GRAPH_PERCENTILE (float): Percentile threshold for building the GNN input graph from raw |Pearson|.
            Lower values (e.g., 50) produce a denser, less biased graph that allows the GNN to discover
            edges that WGCNA would filter out. Distinct from CORR_THRESHOLD_PERCENTILE.
            Set to 75 as a balance between coverage and noise reduction.
        WGCNA_POWER_BETA (int): Soft-thresholding power parameter beta for weighted gene co-expression networks.
        INFERENCE_TARGET_DENSITY (float): Target density percentage for the final inferred network (e.g., 0.05).
        PATIENCE (int): Number of consecutive epochs with no improvement before early stopping triggers.
        MIN_DELTA (float): Minimum improvement in loss required to reset the patience counter.
        N_SEEDS (list[int]): List of random seeds used for multi-seed averaging. Running multiple seeds
            and reporting mean ± std is required for statistical claims in publications.
        GAT_HIDDEN_CHANNELS (int): Dimension of hidden layers in the GATv2 encoder.
        GAT_OUT_CHANNELS (int): Dimension of the output latent space.
        GAT_HEADS (int): Number of parallel attention heads in the first GATv2 layer.
        EPOCHS (int): Total number of training epochs for the GNN.
        LR (float): Learning rate parameter for the Adam optimizer.
        LAMBDA_RECON (float): Loss weight coefficient for network edge reconstruction (BCE).
        LAMBDA_KL (float): Loss weight coefficient for Kullback-Leibler divergence regularization.
            A value of 0.01 enforces a meaningful Gaussian prior on the latent space without
            overwhelming the reconstruction signal.
        LAMBDA_SCALE_FREE (float): Loss weight coefficient for scale-free topological regularization.
        LAMBDA_SPARSITY (float): Loss weight coefficient for targeted sparsity regularization.
        SF_GAMMA (float): Target scale-free power-law exponent.
        EVAL_EDGE_THRESHOLD (float): Threshold percentile score used during metric evaluations.
        GAT_DROPOUT (float): Dropout rate on GATv2 attention coefficients (regularization for small datasets).
        DYNAMIC_NEG_SAMPLING (bool): Re-sample negative edges every training epoch for better generalisation.
        ARACNE_DPI_TOLERANCE (float): DPI tolerance for ARACNE indirect-edge pruning (0.1 = 10% margin).
        PRECISION_AT_K (list[int]): K values for the Precision@K evaluation metric.
    """
    # ── Development / paper mode ──────────────────────────────────────────────
    # FAST_MODE=True  → runs a single seed (~20 min on M4) for rapid iteration.
    # FAST_MODE=False → uses the full N_SEEDS list (set 5+ for paper-quality stats).
    FAST_MODE: bool = False

    SEED = 123

    # ── Modo local (CSV pre-procesados) ──────────────────────────────────────
    # Si LOCAL_MODE=True, se omite la descarga de GEO y se cargan directamente
    # los ficheros counts_control_CLEAN.csv y counts_primary_CLEAN.csv desde
    # LOCAL_DATASET_PATH (directorio del dataset elegido).
    LOCAL_MODE: bool = False
    LOCAL_DATASET_PATH = None   # Path al directorio del dataset seleccionado
    DATASET_NAME: str = ""      # Nombre legible del dataset (p.ej. 'TCGA-BRCA-Basal')
    # ─────────────────────────────────────────────────────────────────────────

    GEO_ID = None
    GEO_METADATA_KEY = None
    GEO_GROUPS = {}       
    GEO_CONTROL_GROUP = None  
    
    CONDITION_COL = "label"
    MIN_TOTAL_COUNTS_PER_GENE = 10
    ALPHA = 0.05
    LOG2FC_THRESH = 0.58       
    N_CPUS = 6  # FIX: raised from 4. M4 has 4 perf + 6 eff cores; 6 avoids thrashing.
    
    BIOGRID_URL = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.227/BIOGRID-ORGANISM-4.4.227.tab3.zip"
    BIOGRID_FILE_MATCH = "BIOGRID-ORGANISM-Homo_sapiens"
    STRING_LINKS_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    STRING_INFO_URL = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
    STRING_SCORE_THRESH = 400 
    
    GENEMANIA_ID_MAPPING_URL = "http://genemania.org/data/current/Homo_sapiens/identifier_mappings.txt"
    GENEMANIA_COEXP_URLS = [
        "http://genemania.org/data/current/Homo_sapiens/Co-expression.Wang-Maris-2006.txt",
        "http://genemania.org/data/current/Homo_sapiens/Co-expression.Mallon-McKay-2013.txt",
        "http://genemania.org/data/current/Homo_sapiens/Co-expression.Roth-Zlotnik-2006.txt"
    ] 
    
    TF_LIST_PATH = str(DATA_DIR / "TF_names_v_1.01.txt")
    TF_BOOST_FACTOR = 1.5      
    
    CORR_THRESHOLD_PERCENTILE = 95   # Used only for the WGCNA baseline binarization
    INPUT_GRAPH_PERCENTILE = 75       # FIX: 75th pct — less noise than 50th, still broader than WGCNA 95th
    WGCNA_POWER_BETA = 4
    INFERENCE_TARGET_DENSITY = 0.05  # Top-K density for ALL binary networks (standardised)
    
    GAT_HIDDEN_CHANNELS = 128   # FIX: raised from 64 — more representational capacity
    GAT_OUT_CHANNELS = 32
    GAT_HEADS = 4               # FIX: raised from 2 — richer multi-head attention
    GAT_DROPOUT: float = 0.2    # Attention dropout on coefficients (regularisation for DEG-sized datasets)
    EPOCHS = 500                # Max epochs; early stopping will halt before if converged
    LR = 0.001
    DYNAMIC_NEG_SAMPLING: bool = True   # Re-sample negative edges each epoch → better generalisation
    PATIENCE = 50               # Early stopping: halt if no improvement for 50 consecutive epochs
    MIN_DELTA = 1e-4            # Early stopping: minimum loss improvement to reset patience
    N_SEEDS = [42, 123, 456]    # Full paper run. Use 5+ seeds for final submission.
                                # Override at runtime: set FAST_MODE=True to use [123] only.
    
    LAMBDA_RECON = 5.0         
    LAMBDA_KL = 0.01            # FIX: raised from 0.0005 — meaningful KL regularization
    LAMBDA_SCALE_FREE = 2.0    
    LAMBDA_SPARSITY = 0.3      
    SF_GAMMA = 2.5

    EVAL_EDGE_THRESHOLD = 0.95
    ARACNE_DPI_TOLERANCE: float = 0.1        # DPI tolerance — now explicit (was getattr default)
    PRECISION_AT_K: list = [50, 100, 200, 500]  # K values for Precision@K evaluation

CFG = Config()

DATA_DIR.mkdir(parents=True, exist_ok=True)
