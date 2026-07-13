#!/bin/bash
#############################################################
#  setup_env_cica.sh
#  Crea el entorno conda 'tagat_env' en el CICA.
#  Ejecutar UNA SOLA VEZ desde un nodo de login (no de cómputo):
#
#    chmod +x setup_env_cica.sh
#    bash setup_env_cica.sh
#############################################################

set -e   # salir si cualquier comando falla

TAGAT_ROOT="$HOME/TA-GAT"
ENV_NAME="tagat_env"

echo "============================================================"
echo "  Creando entorno conda: $ENV_NAME"
echo "  TAGAT_ROOT: $TAGAT_ROOT"
echo "============================================================"

# ── Cargar conda ──────────────────────────────────────────────
module load Anaconda3/2023.09-0   # ajusta al módulo disponible en el CICA
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Crear entorno con Python 3.11 ────────────────────────────
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

# ── Instalar PyTorch con CUDA 12.1 ───────────────────────────
# Consultar https://pytorch.org/get-started/locally/ si cambia la versión CUDA
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# ── Instalar PyTorch Geometric y extensiones ─────────────────
pip install torch-geometric==2.5.3
pip install pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# ── Instalar el resto de dependencias ────────────────────────
pip install -r "$TAGAT_ROOT/requirements.txt"

# ── Verificar instalación ─────────────────────────────────────
echo ""
echo "[CHECK] Verificando instalación..."
python -c "
import torch, torch_geometric
print(f'  PyTorch          : {torch.__version__}')
print(f'  CUDA disponible  : {torch.cuda.is_available()}')
print(f'  PyG              : {torch_geometric.__version__}')
import pydeseq2, GEOparse, networkx, sklearn, scipy
print('  pydeseq2 / GEOparse / networkx / sklearn / scipy : OK')
"

echo ""
echo "============================================================"
echo "  Entorno '$ENV_NAME' creado correctamente."
echo "  Actívalo con:  conda activate $ENV_NAME"
echo "============================================================"
