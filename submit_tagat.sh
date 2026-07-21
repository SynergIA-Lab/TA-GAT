#!/bin/bash
#############################################################
#  TA-GAT — SLURM Batch Script para el Supercomputador CICA
#
#  Uso:
#    # Un solo dataset:
#    sbatch submit_tagat.sh TCGA-BRCA-Basal
#
#    # Todos los datasets en paralelo (array job):
#    sbatch --array=0-3 submit_tagat.sh
#############################################################

# ── Recursos del trabajo (CPU-only: sin cola GPU, acceso más rápido) ───
#SBATCH --job-name=TA-GAT
#SBATCH --output=logs/tagat_%j_%x.out
#SBATCH --error=logs/tagat_%j_%x.err
#SBATCH --time=00:35:00          # 35 min — suficiente para un run CPU completo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # 8 cores CPU (PyDESeq2 + operaciones numpy)
#SBATCH --mem=64G                # 64 GB RAM (sin GPU, todo va a RAM)
#SBATCH --partition=standard     # Partición CPU de CICA (50 nodos idle, sin cola GPU)
##SBATCH --account=TU_CUENTA    # Descomenta si es obligatorio en el CICA

# ── Array de datasets disponibles ────────────────────────────
DATASETS=(
    "TCGA-BRCA-Basal"
    "TCGA-HNSC-Larynx"
    "TCGA-LUAD-LowerLobeLung"
    "TCGA-STAD-CIN"
)

# ── Selección del dataset ─────────────────────────────────────
# Prioridad: argumento $1 > SLURM array task > primer dataset
if [ -n "$1" ]; then
    DATASET_NAME="$1"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    DATASET_NAME="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
else
    DATASET_NAME="${DATASETS[0]}"
fi

echo "============================================================"
echo "  TA-GAT — Inicio de ejecución"
echo "  Dataset : $DATASET_NAME"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Nodo    : $SLURMD_NODENAME"
echo "  Fecha   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── Directorio raíz del proyecto ─────────────────────────────
# SLURM copia el script a /var/spool/slurmd/jobXXX/ al ejecutar,
# por lo que BASH_SOURCE[0] apuntaría a ese directorio temporal.
# SLURM_SUBMIT_DIR siempre contiene el directorio original del sbatch.
TAGAT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "  -> TAGAT_ROOT: $TAGAT_ROOT"
cd "$TAGAT_ROOT" || { echo "[ERROR] No se encontró $TAGAT_ROOT"; exit 1; }

# Crear directorios necesarios
mkdir -p "$TAGAT_ROOT/new_implementation/logs"
mkdir -p "$TAGAT_ROOT/new_implementation/out"

# ── Cargar módulos del sistema ────────────────────────────────
module load Python/3.11.5-GCCcore-13.2.0

# ── Activar entorno virtual venv ──────────────────────────────
if [ -d "$HOME/tagat_env_venv" ]; then
    echo "  -> Activando entorno virtual venv desde $HOME..."
    source "$HOME/tagat_env_venv/bin/activate"
else
    echo "  ❌ ERROR: No se encontró el entorno virtual en $HOME/tagat_env_venv"
    echo "     Ejecuta primero: bash setup_env_cica.sh"
    exit 1
fi

# ── Verificar entorno CPU ─────────────────────────────────────
echo ""
echo "[CPU] Entorno de cómputo:"
python -c "
import torch, os
print(f'  PyTorch  : {torch.__version__}')
print(f'  Device   : CPU (forzado)')
print(f'  Threads  : {torch.get_num_threads()}')
print(f'  CUDA     : {torch.cuda.is_available()} (no se usará)')
"
echo ""

# ── Variables de entorno para el pipeline ────────────────────
export LOCAL_DATASET="$DATASET_NAME"   # activa modo automático (no interactivo)
export PYTHONPATH="$TAGAT_ROOT:$PYTHONPATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""        # Ocultar cualquier GPU — forzar CPU
export PYTORCH_NO_CUDA=1              # Asegurar que PyTorch no intente usar CUDA

# ── Lanzar pipeline ───────────────────────────────────────────
echo "[START] Lanzando TA-GAT para: $DATASET_NAME"
echo ""

python "$TAGAT_ROOT/new_implementation/main.py" \
    2>&1 | tee "$TAGAT_ROOT/new_implementation/logs/tagat_${DATASET_NAME}_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

# ── Resumen final ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  TA-GAT — Fin"
echo "  Dataset   : $DATASET_NAME"
echo "  Exit code : $EXIT_CODE"
echo "  Duración  : $(date -d@$SECONDS -u +%H:%M:%S 2>/dev/null || echo ${SECONDS}s)"
echo "  Fecha fin : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
