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
        CORR_THRESHOLD_PERCENTILE (float): Percentile threshold for binarizing baseline correlation networks (e.g., 95).
        WGCNA_POWER_BETA (int): Soft-thresholding power parameter beta for weighted gene co-expression networks.
        INFERENCE_TARGET_DENSITY (float): Target density percentage for the final inferred network (e.g., 0.05).
        GAT_HIDDEN_CHANNELS (int): Dimension of hidden layers in the GATv2 encoder.
        GAT_OUT_CHANNELS (int): Dimension of the output latent space.
        GAT_HEADS (int): Number of parallel attention heads in the first GATv2 layer.
        EPOCHS (int): Total number of training epochs for the GNN.
        LR (float): Learning rate parameter for the Adam optimizer.
        LAMBDA_RECON (float): Loss weight coefficient for network edge reconstruction (BCE).
        LAMBDA_KL (float): Loss weight coefficient for Kullback-Leibler divergence.
        LAMBDA_SCALE_FREE (float): Loss weight coefficient for scale-free topological regularization.
        LAMBDA_SPARSITY (float): Loss weight coefficient for targeted sparsity regularization.
        SF_GAMMA (float): Target scale-free power-law exponent.
        EVAL_EDGE_THRESHOLD (float): Threshold percentile score used during metric evaluations.
    """
    SEED = 123
    
    GEO_ID = None
    GEO_METADATA_KEY = None
    GEO_GROUPS = {}       
    GEO_CONTROL_GROUP = None  
    
    CONDITION_COL = "label"
    MIN_TOTAL_COUNTS_PER_GENE = 10
    ALPHA = 0.05
    LOG2FC_THRESH = 0.58       
    N_CPUS = 4
    
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
    
    CORR_THRESHOLD_PERCENTILE = 95  
    WGCNA_POWER_BETA = 4            
    INFERENCE_TARGET_DENSITY = 0.05 
    
    GAT_HIDDEN_CHANNELS = 64
    GAT_OUT_CHANNELS = 32
    GAT_HEADS = 2
    EPOCHS = 500
    LR = 0.001
    
    LAMBDA_RECON = 5.0         
    LAMBDA_KL = 0.0005         
    LAMBDA_SCALE_FREE = 2.0    
    LAMBDA_SPARSITY = 0.3      
    SF_GAMMA = 2.5

    EVAL_EDGE_THRESHOLD = 0.95 
    
CFG = Config()

DATA_DIR.mkdir(parents=True, exist_ok=True)
