import os
from pathlib import Path

# Paths principales
BASE_DIR = Path(__file__).resolve().parent.parent
NEW_IMPL_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

class Config:
    SEED = 123
    
    # --- Datos GEO Dinámicos ---
    GEO_ID = None
    GEO_METADATA_KEY = None
    GEO_GROUPS = {}       # e.g. {"Tumor": ["3"], "Normal": ["1"]}
    GEO_CONTROL_GROUP = None  # e.g. "Normal"
    
    # --- PyDESeq2 ---
    CONDITION_COL = "label"
    MIN_TOTAL_COUNTS_PER_GENE = 10
    ALPHA = 0.05
    LOG2FC_THRESH = 0.58       # FC de ~1.5
    N_CPUS = 4
    
    # --- Gold Standard (BioGRID, STRING & GeneMANIA Co-expression) ---
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
    
    # --- Transcription Factors ---
    TF_LIST_PATH = str(DATA_DIR / "TF_names_v_1.01.txt")
    TF_BOOST_FACTOR = 1.5      
    
    # --- Red y Correlaciones Básicas ---
    CORR_THRESHOLD_PERCENTILE = 95  
    WGCNA_POWER_BETA = 4            
    INFERENCE_TARGET_DENSITY = 0.05 
    
    # --- TA-GAT Model (Topology-Aware GNN) ---
    GAT_HIDDEN_CHANNELS = 64
    GAT_OUT_CHANNELS = 32
    GAT_HEADS = 2
    EPOCHS = 500
    LR = 0.001
    
    # Pesos de la Función de Pérdida
    LAMBDA_RECON = 5.0         
    LAMBDA_KL = 0.0005         
    LAMBDA_SCALE_FREE = 2.0    
    LAMBDA_SPARSITY = 0.3      
    SF_GAMMA = 2.5

    # Evaluación
    EVAL_EDGE_THRESHOLD = 0.95 
    
CFG = Config()

# Crear directorio DATA si no existe
DATA_DIR.mkdir(parents=True, exist_ok=True)
