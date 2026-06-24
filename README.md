# GNN Network Inference Project

Este proyecto utiliza Graph Neural Networks (GNNs) para inferir redes de interacción génica a partir de datos de expresión (RNA-Seq/Microarray) integrando conocimientos biológicos y métricas topológicas.

## Requisitos Previos

- Python 3.9 o superior.
- Recomendado: Entorno virtual (venv o conda).

## Instalación

1. Descomprime el archivo.
2. Crea un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Mac/Linux
   # o
   venv\Scripts\activate     # En Windows
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Para ejecutar el pipeline completo (Descarga de datos, DESeq2, entrenamiento GNN e inferencia):

```bash
cd new_implementation
python main.py
```

## Estructura del Proyecto

- `new_implementation/`: Contiene el código fuente principal.
  - `main.py`: Script principal de ejecución.
  - `config.py`: Parámetros de configuración y umbrales.
  - `data_loader.py`: Carga de datos (GEO, BioGRID, STRING) y preprocesamiento.
  - `model_gnn.py`: Arquitectura de la red neuronal (TA-GAT).
  - `metrics.py`: Cálculo de métricas de rendimiento y topología.
  - `functional_analysis.py`: Análisis de enriquecimiento (GO/KEGG).
- `data/`: Directorio para datos de entrada (incluye la lista de TFs).
- `requirements.txt`: Lista de librerías necesarias.

## Notas

El script descargará automáticamente los datasets de GEO y las bases de datos de BioGRID/STRING la primera vez que se ejecute si no están presentes en la carpeta `data/`.


TA-GAT Link: https://nas.synergialab.eu/s/ZEf2WWnmmbKfKaF
