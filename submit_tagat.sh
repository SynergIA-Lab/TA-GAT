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

# ── Recursos del trabajo (Optimizado para entrar rápido en cola) ────
#SBATCH --job-name=TA-GAT
#SBATCH --output=logs/tagat_%j_%x.out
#SBATCH --error=logs/tagat_%j_%x.err
#SBATCH --time=03:00:00          # Reducido a 3h (suficiente para run completo de 3 seeds)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Reducido a 4 cores (prioridad alta)
#SBATCH --mem=32G                # Reducido a 32G (suficiente para DESeq2)
#SBATCH --gres=gpu:1             # 1 GPU para aceleración GATv2
#SBATCH --partition=gpu          # Partición GPU de Hércules
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

# ── Directorio raíz del proyecto (auto-detectado desde la ubicación del script) ──
TAGAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# ── Verificar GPU disponible ──────────────────────────────────
echo ""
echo "[GPU] Dispositivos disponibles:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi no disponible)"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA OK  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM     : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
"
echo ""

# ── Variables de entorno para el pipeline ────────────────────
export LOCAL_DATASET="$DATASET_NAME"   # activa modo automático (no interactivo)
export PYTHONPATH="$TAGAT_ROOT:$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

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
