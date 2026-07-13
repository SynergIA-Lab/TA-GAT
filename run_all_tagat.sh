#!/bin/bash
#############################################################
#  run_all_tagat.sh  —  Lanzador principal de TA-GAT
#
#  Ejecuta los 4 datasets TCGA en paralelo en el cluster CICA.
#  Solo necesitas ejecutar ESTE script:
#
#    bash run_all_tagat.sh
#
#############################################################

TAGAT_ROOT="$HOME/TA-GAT"

# ── Verificar que existen los 4 datasets ─────────────────────
DATASETS=(
    "TCGA-BRCA-Basal"
    "TCGA-HNSC-Larynx"
    "TCGA-LUAD-LowerLobeLung"
    "TCGA-STAD-CIN"
)

echo "============================================================"
echo "  TA-GAT — Verificando datasets antes de lanzar"
echo "============================================================"
ALL_OK=true
for DS in "${DATASETS[@]}"; do
    CTRL="$TAGAT_ROOT/data/Conjuntos a ejecutar/$DS/counts_control_CLEAN.csv"
    PRIM="$TAGAT_ROOT/data/Conjuntos a ejecutar/$DS/counts_primary_CLEAN.csv"
    if [ -f "$CTRL" ] && [ -f "$PRIM" ]; then
        echo "  ✅  $DS"
    else
        echo "  ❌  $DS  (faltan archivos CSV)"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "[ERROR] Faltan archivos en algún dataset. Comprueba la subida."
    exit 1
fi

# ── Crear directorio de logs ─────────────────────────────────
mkdir -p "$TAGAT_ROOT/new_implementation/logs"

echo ""
echo "============================================================"
echo "  Lanzando los 4 datasets como Array Job..."
echo "============================================================"

# ── Lanzar los 4 en paralelo (un job por dataset) ────────────
JOB_ID=$(sbatch --array=0-3 "$TAGAT_ROOT/submit_tagat.sh" | awk '{print $NF}')

echo ""
echo "  Job array enviado: ID = $JOB_ID"
echo "  Datasets asignados:"
echo "    [0] TCGA-BRCA-Basal          → job ${JOB_ID}_0"
echo "    [1] TCGA-HNSC-Larynx         → job ${JOB_ID}_1"
echo "    [2] TCGA-LUAD-LowerLobeLung  → job ${JOB_ID}_2"
echo "    [3] TCGA-STAD-CIN            → job ${JOB_ID}_3"
echo ""
echo "  Monitorizar:"
echo "    squeue -u \$USER"
echo "    tail -f $TAGAT_ROOT/new_implementation/logs/tagat_TCGA-BRCA-Basal_${JOB_ID}.log"
echo ""
echo "  Cancelar todo si hay algún problema:"
echo "    scancel $JOB_ID"
echo "============================================================"
