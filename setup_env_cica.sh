#!/bin/bash
#############################################################
#  setup_env_cica.sh  —  Crea el entorno de ejecución
#
#  Configurado para el módulo Python verificado de Hércules (CICA).
#  Ejecutar en el nodo de login:  bash setup_env_cica.sh
#############################################################

set -e

TAGAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HOME/tagat_env_venv"

echo "============================================================"
echo "  TA-GAT — Configuración del entorno (CICA)"
echo "  TAGAT_ROOT : $TAGAT_ROOT"
echo "  VENV_DIR   : $ENV_DIR"
echo "============================================================"
echo ""

# 1. Cargar el módulo Python verificado del CICA
echo "[1/4] Cargando módulo Python/3.11.5-GCCcore-13.2.0..."
module load Python/3.11.5-GCCcore-13.2.0

echo "  Usando: $(python3 --version) de $(which python3)"
echo ""

# 2. Crear entorno virtual estándar en tu HOME
echo "[2/4] Creando Entorno Virtual (venv) en: $ENV_DIR..."
if [ -d "$ENV_DIR" ]; then
    echo "  ⚠️ El entorno venv ya existe. Reutilizando..."
else
    python3 -m venv "$ENV_DIR"
    echo "  ✅ Entorno virtual creado."
fi

# Activar el entorno virtual
source "$ENV_DIR/bin/activate"
echo "  ✅ Entorno virtual activado."
echo ""

# 3. Actualizar pip e instalar PyTorch + PyG con soporte CUDA
echo "[3/4] Instalando PyTorch + PyG (CUDA 12.1 compatible)..."
pip install --upgrade pip

# PyTorch 2.3.1 con CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# PyG y extensiones sparse compatibles con torch 2.3.1 + cu121
pip install torch-geometric==2.5.3
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

echo "  ✅ Librerías de Deep Learning instaladas."
echo ""

# 4. Instalar el resto de dependencias
echo "[4/4] Instalando dependencias del proyecto..."
pip install -r "$TAGAT_ROOT/requirements.txt"
echo ""

# ── Verificación final ────────────────────────────────────────
echo "─────────────────────────────────────────────────────────"
echo "  VERIFICACIÓN FINAL"
echo "─────────────────────────────────────────────────────────"
python - <<'PYCHECK'
import sys
print(f"  Python       : {sys.version.split()[0]}")
import torch
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA OK      : {torch.cuda.is_available()}")
import torch_geometric
print(f"  PyG          : {torch_geometric.__version__}")
PYCHECK

echo ""
echo "============================================================"
echo "  Configuración completada con éxito."
echo "  Entorno virtual listo en: $ENV_DIR"
echo "  Para activarlo manualmente:"
echo "    source $ENV_DIR/bin/activate"
echo "============================================================"
