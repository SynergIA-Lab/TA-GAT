import os
import GEOparse
from pathlib import Path
from config import CFG, DATA_DIR, NEW_IMPL_DIR

##################################################
# PHASE 1: GEO METADATA PARSING & EXTRACTION    #
##################################################

def parse_geo_metadata(gse: GEOparse.GSE) -> dict[str, list[str]]:
    """
    Parses structural metadata fields and their unique values from GEO GSM objects.

    Iterates through the sample records within the GEO dataset, extracts key-value 
    pairs from the 'characteristics_ch1' metadata, and aggregates the unique clinical 
    or experimental values available for cohort partitioning.

    Args:
        gse (GEOparse.GSE): The GSE object representing the downloaded GEO study.

    Returns:
        dict[str, list[str]]: A dictionary mapping metadata property keys to a list 
            of their unique available categorical values.
    """
    props = {}
    for gsm_name, gsm in gse.gsms.items():
        chars = gsm.metadata.get("characteristics_ch1", [])
        for c in chars:
            if ":" in c:
                key, val = c.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key not in props:
                    props[key] = set()
                props[key].add(val)
            else:
                key = "characteristics_otras"
                if key not in props:
                    props[key] = set()
                props[key].add(c.strip())
    return {k: list(v) for k, v in props.items() if len(v) > 0}

##################################################
# PHASE 2: INTERACTIVE COHORT CONFIGURATION      #
##################################################

def run_interactive_setup() -> tuple[Path, Path]:
    """
    Launches an interactive command-line setup wizard to configure the GEO dataset.

    Prompts the user to input a GSE Accession ID, downloads the corresponding dataset, 
    displays all detected metadata properties, and guides the user in selecting 
    the grouping variable. It then prompts for group assignment values to split 
    samples into 'Normal' (reference control) and 'Tumor' (case) cohorts.
    Creates and returns output and figure directories.

    Returns:
        tuple[Path, Path]: A tuple containing:
            - out_dir (Path): The output directory path for saving execution logs and TSV matrices.
            - figures_dir (Path): The directory path for saving generated network figures.
    """
    if os.environ.get("AUTOMATED") == "1":
        CFG.GEO_ID = "GSE11121"
        out_dir = NEW_IMPL_DIR / "out" / CFG.GEO_ID
        figures_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        CFG.GEO_METADATA_KEY = "grade"
        CFG.GEO_GROUPS = {"Tumor": ["3"], "Normal": ["1"]}
        CFG.GEO_CONTROL_GROUP = "Normal"
        print("[AUTOMATED] Running in automated mode with GSE11121, splitting by 'grade' (Tumor: [3], Normal: [1])")
        return out_dir, figures_dir

    print("=" * 60)
    print("   CONFIGURACIÓN INTERACTIVA DEL CONJUNTO DE DATOS (GEO)   ")
    print("=" * 60)
    
    geo_id = input("\n1. Introduce el ID del dataset de GEO (por ejemplo, GSE11121):\n> ").strip()
    if not geo_id:
        print("  -> No se introdujo ID, usando GSE11121 por defecto.")
        geo_id = "GSE11121"
    
    CFG.GEO_ID = geo_id
    
    out_dir = NEW_IMPL_DIR / "out" / CFG.GEO_ID
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDescargando y analizando metadatos para {geo_id}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gse = GEOparse.get_GEO(geo=geo_id, destdir=str(DATA_DIR), silent=True)
    
    props = parse_geo_metadata(gse)
    if not props:
        print("\n[ERROR] No se encontraron metadatos estructurados en characteristics_ch1.")
        print("Continuaremos con valores por defecto (grade -> Tumor/Normal) si aplica.\n")
        CFG.GEO_METADATA_KEY = "grade"
        CFG.GEO_GROUPS = {"Tumor": ["3"], "Normal": ["1"]}
        CFG.GEO_CONTROL_GROUP = "Normal"
        return out_dir, figures_dir
        
    print("\nPROPIEDADES DISPONIBLES EN LOS METADATOS:")
    keys_list = list(props.keys())
    for idx, k in enumerate(keys_list):
        print(f"  [{idx + 1}] {k} -> Valores: {props[k]}")
        
    prop_idx_input = input("\n2. Selecciona el número de la propiedad qué quieres usar para separar los datos:\n> ").strip()
    try:
        prop_idx = int(prop_idx_input) - 1
        prop_selected = keys_list[prop_idx]
    except (ValueError, IndexError):
        print("  -> Entrada inválida. Usando la primera propiedad por defecto.")
        prop_selected = keys_list[0]
        
    print(f"\n> Has seleccionado separar por: '{prop_selected}'\n")
    CFG.GEO_METADATA_KEY = prop_selected
    
    print(f"\n3. Asignación de muestras a grupos. Valores disponibles: {props[prop_selected]}")
    groups_dict = {}
    
    print(f"\n--- Grupo 'Normal' (Control/Referencia) ---")
    g_vals_normal = input("¿Qué valores pertenecen a 'Normal'? (Escribe los valores separados por coma, exactos como aparecen arriba):\n> ").strip()
    groups_dict["Normal"] = [v.strip() for v in g_vals_normal.split(",") if v.strip()]
    
    print(f"\n--- Grupo 'Tumor' (Caso) ---")
    g_vals_tumor = input("¿Qué valores pertenecen a 'Tumor'? (Escribe los valores separados por coma, exactos como aparecen arriba):\n> ").strip()
    groups_dict["Tumor"] = [v.strip() for v in g_vals_tumor.split(",") if v.strip()]
        
    CFG.GEO_GROUPS = groups_dict
    CFG.GEO_CONTROL_GROUP = "Normal"
    
    print("\nGrupos definidos:")
    for k, v in groups_dict.items():
        print(f"  - {k}: {v}")
    print("\n¡El grupo 'Normal' será la base de referencia (Control) para extraer DEGs!")
    
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN FINALIZADA. PROCEDIENDO AL PIPELINE...")
    print("=" * 60 + "\n")
    
    return out_dir, figures_dir
