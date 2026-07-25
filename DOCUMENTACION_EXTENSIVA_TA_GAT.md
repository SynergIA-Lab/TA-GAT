# Documentación Extensiva — TA-GAT
## Topology-Aware Graph Attention Network para Inferencia de Redes de Regulación Génica

> **Propósito de este documento**: Explicar de forma exhaustiva, paso a paso y sin asumir conocimiento previo del código, qué hace el algoritmo TA-GAT desde el instante en que se lanza hasta que termina. Cubre tipos de datos, estructuras internas, parámetros, arquitectura neuronal, funciones de pérdida y métricas de evaluación.

---

## Índice

1. [Visión Global del Pipeline](#1-visión-global-del-pipeline)
2. [Módulo de Configuración — config.py](#2-módulo-de-configuración--configpy)
3. [Configuración Interactiva — interactive_setup.py](#3-configuración-interactiva--interactive_setuppy)
4. [Carga de Datos y Preprocesamiento — data_loader.py](#4-carga-de-datos-y-preprocesamiento--data_loaderpy)
5. [Arquitectura del Modelo GNN — model_gnn.py](#5-arquitectura-del-modelo-gnn--model_gnnpy)
6. [Entrenamiento del Modelo — main.py](#6-entrenamiento-del-modelo--mainpy)
7. [Inferencia de Red](#7-inferencia-de-red)
8. [Evaluación y Métricas — metrics.py](#8-evaluación-y-métricas--metricspy)
9. [Análisis de Relevancia Biológica — biological_relevance.py](#9-análisis-de-relevancia-biológica--biological_relevancepy)
10. [Salidas del Sistema](#10-salidas-del-sistema)
11. [Flujo Completo — Diagrama Paso a Paso](#11-flujo-completo--diagrama-paso-a-paso)

---

## 1. Visión Global del Pipeline

TA-GAT es un sistema de **inferencia de redes de regulación génica (GRN)** basado en un **Autoencoder Variacional de Grafos con Atención** (Graph Attention Variational Autoencoder). El objetivo es descubrir qué genes interactúan funcionalmente a partir de datos de expresión génica de RNA-seq, superando a los métodos clásicos de correlación (Pearson, Spearman, WGCNA, ARACNE).

### Orden de ejecución de alto nivel

```
main() -> interactive_setup -> carga TFs -> carga expresión + DESeq2
       -> descarga Gold Standards (BioGRID, STRING, GeneMANIA)
       -> por cada condición experimental:
           -> multi-seed loop:
               -> build_correlation_baseline  (grafo PyG + baselines)
               -> entrenamiento TA-GAT         (GATv2 VAE + 4 pérdidas)
               -> infer_network                (cosine-similarity top-K)
               -> evaluate_against_gold_standard (AUC, AUPRC, P@K, F1...)
           -> ensemble de semillas
           -> Wilcoxon pairwise tests
       -> print_final_summary_tables
       -> analyze_hubs + compute_differential_network
```

**Punto de entrada**: `new_implementation/main.py`, función `main()`.

---

## 2. Módulo de Configuración — `config.py`

### 2.1 Clase `Config`

Todos los hiperparámetros, rutas y URLs del sistema están centralizados en la clase `Config` (instanciada globalmente como `CFG`). No hay ningún parámetro mágico disperso por el código: todo viene de aquí.

#### Rutas del sistema

| Variable | Tipo | Descripción |
|---|---|---|
| `BASE_DIR` | `Path` | Directorio raíz del proyecto (dos niveles sobre `config.py`) |
| `NEW_IMPL_DIR` | `Path` | Directorio del código fuente (`BASE_DIR/new_implementation`) |
| `DATA_DIR` | `Path` | Directorio de datos locales (`BASE_DIR/data`) |

#### Modos de ejecución

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `FAST_MODE` | `bool` | `False` | Si `True`, usa solo 1 semilla (`[123]`) para desarrollo rápido (~20 min). Para el paper debe ser `False` |
| `LOCAL_MODE` | `bool` | `False` | Si `True`, carga CSVs locales en lugar de descargar de GEO |
| `LOCAL_DATASET_PATH` | `Path` | `None` | Ruta al directorio del dataset local seleccionado |
| `DATASET_NAME` | `str` | `""` | Nombre legible del dataset (ej. `TCGA-BRCA-Basal`) |

#### Configuración del dataset GEO

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `GEO_ID` | `str` | `None` | ID de acceso a GEO (ej. `GSE11121`) |
| `GEO_METADATA_KEY` | `str` | `None` | Columna de metadatos para separar cohortes (ej. `grade`) |
| `GEO_GROUPS` | `dict` | `{}` | Mapeo de nombre de grupo a lista de valores del metadato |
| `GEO_CONTROL_GROUP` | `str` | `None` | Grupo de referencia para PyDESeq2 (típicamente `Normal`) |
| `CONDITION_COL` | `str` | `label` | Nombre de la columna de diseño experimental |

#### Parámetros de filtrado de expresión y DESeq2

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `MIN_TOTAL_COUNTS_PER_GENE` | `int` | `10` | Suma mínima de cuentas por gen para no ser filtrado |
| `ALPHA` | `float` | `0.05` | Nivel de significancia para FDR (valor-p ajustado) |
| `LOG2FC_THRESH` | `float` | `0.58` | Umbral mínimo de `|log2FoldChange|` (~1.5× cambio de expresión) |
| `N_CPUS` | `int` | `4` | Núcleos paralelos para PyDESeq2 |

#### URLs de bases de datos de referencia

| Parámetro | Descripción |
|---|---|
| `BIOGRID_URL` | Interacciones físicas/genéticas humanas validadas experimentalmente (BioGRID 4.4.227) |
| `STRING_LINKS_URL` | Red de interacciones proteína-proteína STRING v12 (Homo sapiens) |
| `STRING_INFO_URL` | Mapeo ENSP → símbolo HUGO para STRING v12 |
| `STRING_SCORE_THRESH = 400` | Puntuación mínima de confianza STRING (escala 0–1000) |
| `GENEMANIA_ID_MAPPING_URL` | Mapeo de identificadores GeneMANIA |
| `GENEMANIA_COEXP_URLS` | 3 redes de co-expresión humanas GeneMANIA (Wang-2006, Mallon-2013, Roth-2006) |

#### Parámetros de correlación y grafo de entrada

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `TF_LIST_PATH` | `str` | `data/TF_names_v_1.01.txt` | Ruta a la lista de factores de transcripción conocidos |
| `TF_BOOST_FACTOR` | `float` | `1.5` | Factor multiplicativo sobre correlaciones que implican TFs |
| `CORR_THRESHOLD_PERCENTILE` | `float` | `95` | Percentil para binarizar solo la red WGCNA baseline |
| `INPUT_GRAPH_PERCENTILE` | `float` | `75` | Percentil de Pearson para construir el grafo de entrada del GNN |
| `WGCNA_POWER_BETA` | `int` | `4` | Potencia de soft-thresholding de WGCNA (amplifica correlaciones fuertes) |
| `INFERENCE_TARGET_DENSITY` | `float` | `0.05` | Densidad objetivo (5%) para binarizar todas las redes predichas |

#### Hiperparámetros de la arquitectura GNN

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `GAT_HIDDEN_CHANNELS` | `int` | `128` | Dimensión de las capas ocultas GATv2 |
| `GAT_OUT_CHANNELS` | `int` | `32` | Dimensión del espacio latente (mu y log_sigma) |
| `GAT_HEADS` | `int` | `4` | Cabezas de atención paralelas en GATv2 |
| `GAT_DROPOUT` | `float` | `0.2` | Dropout sobre coeficientes de atención (regularización) |

#### Hiperparámetros de entrenamiento

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `EPOCHS` | `int` | `500` | Máximo de épocas; el early stopping puede detenerlo antes |
| `LR` | `float` | `0.001` | Tasa de aprendizaje para el optimizador Adam |
| `PATIENCE` | `int` | `50` | Épocas consecutivas sin mejora antes del early stopping |
| `MIN_DELTA` | `float` | `1e-4` | Mejora mínima de loss para resetear el contador de paciencia |
| `DYNAMIC_NEG_SAMPLING` | `bool` | `True` | Re-muestrear aristas negativas cada época (mejor generalización) |
| `N_SEEDS` | `list[int]` | `[42, 123, 456]` | Semillas aleatorias para el promedio multi-semilla |

#### Pesos de la función de pérdida multi-objetivo

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `LAMBDA_RECON` | `float` | `5.0` | Peso de la pérdida de reconstrucción (Binary Cross-Entropy) |
| `LAMBDA_KL` | `float` | `0.01` | Peso de la divergencia KL (regularización prior gaussiano) |
| `LAMBDA_SCALE_FREE` | `float` | `2.0` | Peso de la regularización de topología scale-free |
| `LAMBDA_SPARSITY` | `float` | `0.3` | Peso de la penalización de esparcidad dirigida |
| `SF_GAMMA` | `float` | `2.5` | Exponente objetivo de la ley de potencia (gamma de Barabási-Albert) |

#### Parámetros de evaluación

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `EVAL_EDGE_THRESHOLD` | `float` | `0.95` | Percentil de score usado durante métricas de evaluación |
| `ARACNE_DPI_TOLERANCE` | `float` | `0.1` | Tolerancia DPI de ARACNE (10% de margen) |
| `PRECISION_AT_K` | `list[int]` | `[50, 100, 200, 500]` | Valores de K para la métrica Precision@K |

---

## 3. Configuración Interactiva — `interactive_setup.py`

### 3.1 Función `run_interactive_setup()`

Primera función que llama `main()`. Determina **cómo se van a proporcionar los datos** al pipeline.

#### Modos de ejecución disponibles

**Modo automático GEO** (`AUTOMATED=1` en variables de entorno):
Configura `CFG.GEO_ID = "GSE11121"` con partición por `grade` (Tumor=3, Normal=1). Útil para CI/CD o clúster.

**Modo automático local** (`LOCAL_DATASET=<nombre_carpeta>` en entorno):
Llama directamente a `run_local_setup()` sin intervención humana.

**Modo interactivo** (sin variables de entorno):
Presenta un menú de dos opciones:
1. **Dataset local pre-procesado**: Escanea `data/Conjuntos a ejecutar/` buscando subdirectorios con `counts_control_CLEAN.csv` y `counts_primary_CLEAN.csv`.
2. **Descarga desde GEO**: Pide el ID de acceso GEO, descarga metadatos y guía al usuario para asignar los valores del metadato a los grupos "Tumor" y "Normal".

#### Salidas

Devuelve `tuple[Path, Path]`:
- `out_dir`: Directorio de salida para logs y matrices TSV (ej. `new_implementation/out/GSE11121/`)
- `figures_dir`: Subdirectorio para figuras (ej. `new_implementation/out/GSE11121/figures/`)

Ambos se crean automáticamente con `mkdir(parents=True, exist_ok=True)`.

### 3.2 Función `run_local_setup()`

Escanea `data/Conjuntos a ejecutar/` y lista los subdirectorios válidos. Configura:
- `CFG.LOCAL_MODE = True`
- `CFG.LOCAL_DATASET_PATH = chosen` (directorio elegido)
- `CFG.DATASET_NAME = chosen.name`
- `CFG.GEO_ID = chosen.name` (se usa como nombre de carpeta de salida)

### 3.3 Función `parse_geo_metadata(gse)`

Itera sobre los objetos GSM del GEO study descargado y extrae los pares clave-valor del campo `characteristics_ch1`. Devuelve `dict[str, list[str]]` con cada propiedad clínica mapeada a sus valores únicos disponibles.

---

## 4. Carga de Datos y Preprocesamiento — `data_loader.py`

### 4.1 Descarga y Parseo de Gold Standards

Los Gold Standards son bases de datos externas de interacciones génicas/proteicas validadas experimentalmente. Se usan **exclusivamente para evaluación**, nunca para entrenar el modelo.

#### 4.1.1 BioGRID — `download_and_parse_biogrid()`

**Qué es**: Base de datos de interacciones físicas y genéticas validadas experimentalmente (co-IP, Y2H, etc.).

**Proceso paso a paso**:
1. Comprueba si el archivo `data/BIOGRID-ORGANISM-Homo_sapiens-4.4.227.tab3.txt` ya existe localmente. Si no, descarga el ZIP (~40 MB).
2. Extrae únicamente el fichero `Homo_sapiens` del ZIP usando `ZipFile.extract()`.
3. Lee el archivo con `pd.read_csv(sep='\t')` seleccionando `usecols=[7, 8, 11, 15, 16]`:
   - Columnas 7 y 8: símbolos de gen A y gen B
   - Columnas 15 y 16: TaxID del organismo A y B (filtro: ambos deben ser 9606 = Homo sapiens)
4. Filtra para que ambos TaxID sean 9606 (elimina interacciones entre organismos distintos).
5. Normaliza símbolos a mayúsculas con `str.upper()`.
6. Elimina self-loops (gen A == gen B).
7. **Deduplicación de aristas no dirigidas**: para cada par `(A, B)`, ordena alfabéticamente y genera `(min(A,B), max(A,B))`. Almacena en un `set` para eliminar duplicados.

**Tipo de dato devuelto**: `set[tuple[str, str]]` — pares únicos donde `node1 < node2` lexicográficamente.
**Tamaño típico**: ~600.000–700.000 aristas únicas humanas.

#### 4.1.2 STRING — `download_and_parse_string()`

**Qué es**: Red de interacciones proteína-proteína (PPI) integrada con puntuaciones de confianza 0–1000.

**Proceso paso a paso**:
1. Descarga `9606.protein.links.v12.0.txt.gz` (~80 MB) y `9606.protein.info.v12.0.txt.gz` si no existen.
2. Parsea el archivo de información para construir `protein_to_gene: dict[str, str]`, que mapea identificadores ENSP (ej. `9606.ENSP00000000001`) a símbolos HUGO (ej. `ARF5`).
3. Lee el archivo de links línea a línea en formato gzip. Para cada triplete `(proteína1, proteína2, score)`:
   - Solo conserva pares con `score >= STRING_SCORE_THRESH` (400 por defecto = confianza media).
   - Traduce ENSP a símbolo de gen usando `protein_to_gene`.
   - Descarta self-loops y añade el par ordenado al conjunto.

**Tipo de dato devuelto**: `set[tuple[str, str]]`.

**Nota sobre el threshold 400**: Es el estándar de la literatura GRN. Incluye evidencia experimental, co-expresión, co-ocurrencia textual y transferencia filogenética.

#### 4.1.3 GeneMANIA — `download_and_parse_genemania_coexp()`

**Qué es**: Red de co-expresión funcional humana basada en 3 estudios de referencia (Wang-Maris-2006, Mallon-McKay-2013, Roth-Zlotnik-2006).

**Proceso paso a paso**:
1. Descarga `genemania_identifier_mappings.txt` (~11 MB). Parsea el mapeo `Preferred_Name → Gene Name` (solo entradas donde `source == "Gene Name"`), generando `pm_to_gene: dict[str, str]`.
2. Para cada URL de red de co-expresión de `CFG.GENEMANIA_COEXP_URLS`:
   - Descarga el archivo si no existe localmente.
   - Lee línea a línea: `gen1\tgen2\tscore` (el score no se usa; basta la presencia del par).
   - Traduce nombres usando `pm_to_gene`, descarta self-loops, añade al conjunto ordenado.

**Tipo de dato devuelto**: `set[tuple[str, str]]`.

#### 4.1.4 Construcción de la Matriz de Adyacencia Gold Standard — `create_ground_truth_adj()`

**Propósito**: Proyectar un conjunto global de interacciones (que puede abarcar todo el genoma) al subconjunto local de DEGs del experimento.

**Entradas**:
- `genes: list[str]` — Lista de DEGs del experimento (máx. 1500).
- `gs_edges: set[tuple[str, str]]` — Conjunto de aristas del Gold Standard.

**Proceso**:
1. Extrae todos los genes del Gold Standard: `gold_genes = set(u for u,v in gs_edges) | set(v for u,v in gs_edges)`.
2. Calcula el **conjunto evaluable**: `evaluable_genes_set = gold_genes ∩ set(genes)`. Solo los genes que aparecen en ambas fuentes pueden ser evaluados de forma justa.
3. Construye `gene_map_idx: dict[str, int]` (gen → índice de fila/columna).
4. Inicializa `adj = np.zeros((N, N), dtype=np.float32)`.
5. Para cada arista `(u, v)` del GS, si ambos genes están en `gene_map_idx`, pone `adj[u,v] = adj[v,u] = 1.0` (matriz simétrica binaria).

**Tipo de dato devuelto**: `tuple[np.ndarray (N×N float32), set[str]]`.

---

### 4.2 Carga de Datos de Expresión Génica y PyDESeq2

#### 4.2.1 Modo GEO — `load_geo_and_run_deseq2()`

**Propósito**: Descargar un dataset de GEO, construir la matriz de expresión e identificar genes diferencialmente expresados (DEGs).

**Proceso detallado**:

**Paso 1 — Descarga del dataset GEO**
Usa `GEOparse.get_GEO(geo=geo_id, destdir=str(destdir), silent=True)` para descargar el study. GEOparse gestiona automáticamente los archivos SOFT/MINiML en `DATA_DIR`.

**Paso 2 — Extracción de expresión y metadatos**
Itera sobre todos los objetos `GSM` (muestras) del study:
- Si `gsm.table` no está vacío y tiene columna `VALUE`, extrae `pd.Series` indexada por `ID_REF` (identificador de sonda/gen).
- Extrae `characteristics_ch1` para buscar la clave `CFG.GEO_METADATA_KEY` y asignar la muestra al grupo correspondiente en `CFG.GEO_GROUPS`.
- Las muestras sin grupo definido se etiquetan como `"discard"` y se excluyen del análisis.

**Paso 3 — Gestión de matrices vacías**
Si las tablas GSM están vacías (frecuente en RNA-seq moderno que almacena la matriz como archivo suplementario):
- Busca en `gse.metadata` archivos suplementarios con extensiones `.csv.gz`, `.tsv.gz`, `.txt.gz`, etc.
- Descarga el primer archivo válido encontrado.
- **Auto-transposición**: Si la matriz tiene más columnas que filas (indica formato `muestras × genes`), la transpone.
- **Mapeo de columnas**: Intenta mapear los nombres de columna del suplementario a los IDs GSM usando los metadatos de cada GSM.
- **Detección y traducción de Ensembl IDs**: Si los índices comienzan por `ENSG`, llama a la API de MyGene.info en chunks de 1000 genes para traducir a símbolos HUGO. Los genes sin mapeo conservan su ID Ensembl.

**Paso 4 — Mapeo GPL (para microarrays)**
Si el study tiene plataforma GPL con columna `Gene Symbol` en su tabla, renombra el índice de la matriz y agrupa sondas del mismo gen con `groupby().mean()`.

**Paso 5 — PyDESeq2: Análisis de Expresión Diferencial**

PyDESeq2 es la implementación Python del algoritmo DESeq2 (Love et al., 2014), diseñado para RNA-seq con cuentas enteras siguiendo una distribución binomial negativa.

Preprocesamiento previo a PyDESeq2:
- Convierte a numérico y rellena NaN con 0.
- **Detección de log-transform**: Si hay valores negativos (indicativo de datos ya en escala logarítmica, ej. `log2(TPM+1)`), aplica `2^x` para revertir a cuentas brutas aproximadas.
- Redondea a enteros (`round().astype(int)`).
- Filtra genes con suma total de cuentas < `MIN_TOTAL_COUNTS_PER_GENE` (por defecto 10). Genes con muy pocas cuentas son ruido.
- **Transposición**: PyDESeq2 espera `counts` en formato `(muestras × genes)` → `counts_sxg = counts_clean.T`.

Inicialización y ajuste del modelo DESeq2:
```python
dds = DeseqDataSet(
    counts=counts_sxg,            # DataFrame (muestras x genes), enteros
    metadata=meta_df,             # columna 'label' con grupo de cada muestra
    design="~ label",             # modelo lineal: diferencia entre grupos
    refit_cooks=True,             # reajusta outliers Cook's distance
    inference=DefaultInference(n_cpus=CFG.N_CPUS),
)
dds.deseq2()
```

`dds.deseq2()` ejecuta internamente:
1. **Estimación de factores de tamaño** (size factors): normalización basada en la mediana de ratios para corregir diferencias de profundidad de secuenciación entre muestras.
2. **Estimación de dispersiones**: ajuste de un modelo de dispersión gen-específico usando máxima verosimilitud local (MAP estimation).
3. **Ajuste del modelo GLM**: regresión binomial negativa generalizada con el diseño especificado.
4. **Prueba de Wald**: para cada gen, contrasta el coeficiente estimado (log2FoldChange) contra la hipótesis nula de ningún cambio.

Para cada contraste `case_group vs control_group`:
```python
stat_res = DeseqStats(
    dds,
    contrast=("label", case_group, control_group),
    inference=inference,
    alpha=CFG.ALPHA,
)
stat_res.summary()
results_df = stat_res.results_df
```

Filtrado de DEGs significativos:
- `padj` (p-valor ajustado por corrección Benjamini-Hochberg/FDR) ≤ `ALPHA = 0.05`.
- `|log2FoldChange|` ≥ `LOG2FC_THRESH = 0.58` (corresponde a ~1.5× de cambio).

**Cap de DEGs**: Si el número de DEGs supera 1500, se toman los 1500 con menor `padj` (más significativos). El límite en 1500 está motivado por la memoria: una red de 1500 nodos tiene 1.500 × (1.500-1) / 2 = 1.124.250 posibles aristas.

**Salidas de `load_geo_and_run_deseq2()`**:
- `combined_degs: pd.DataFrame` — Tabla de resultados PyDESeq2 indexada por símbolo de gen. Columnas: `log2FoldChange`, `pvalue`, `padj`, `baseMean`, `lfcSE`, `stat`.
- `expr_groups_dict: dict[str, pd.DataFrame]` — `{nombre_grupo: DataFrame(num_degs, num_muestras_del_grupo)}`.

#### 4.2.2 Modo Local — `load_local_counts_and_run_deseq2()`

Equivalente al modo GEO pero carga directamente dos CSVs pre-procesados:
- `counts_control_CLEAN.csv` — Matriz `genes × muestras control` (cuentas enteras brutas)
- `counts_primary_CLEAN.csv` — Matriz `genes × muestras tumor/primario` (cuentas enteras brutas)

Los une en una única matriz, asigna etiquetas `"Normal"` y `"Tumor"` a las muestras, y ejecuta el mismo pipeline PyDESeq2 contrastando `Tumor vs Normal`. Configura al final `CFG.GEO_GROUPS = {"Normal": [...], "Tumor": [...]}` para que el resto del pipeline los lea correctamente.

---

### 4.3 Construcción del Grafo de Entrada para el GNN — `build_correlation_baseline()`

Esta es la función más compleja del módulo. Construye tanto los baselines estadísticos como el grafo PyG de entrada al GNN.

**Entradas**:
- `expr: pd.DataFrame (num_genes, num_samples)` — Matriz de expresión para la condición actual.
- `tf_set: set[str]` — Conjunto de factores de transcripción conocidos.
- `degs_df: pd.DataFrame` — Resultados de PyDESeq2 con `log2FoldChange` y `padj`.

#### Paso 1 — Matrices de correlación brutas

Pearson (correlación lineal):
```python
corr_p_vals = np.corrcoef(expr.values)  # (N, N)
corr_p = np.abs(corr_p_vals)            # valores absolutos: activacion y represion tratados igual
np.fill_diagonal(corr_p, 0.0)           # no self-loops
```

Spearman (correlación por rangos, más robusta a valores extremos):
```python
# Aproximacion eficiente via doble argsort:
ranks = np.argsort(np.argsort(expr.values, axis=1), axis=1).astype(float)
corr_s_vals = np.corrcoef(ranks)
corr_s = np.abs(corr_s_vals)
np.fill_diagonal(corr_s, 0.0)
```

Ambas matrices se tratan con `np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` para manejar genes constantes.

#### Paso 2 — WGCNA soft-thresholding

```python
corr_p_powered = corr_p ** beta   # beta = WGCNA_POWER_BETA = 4
corr_s_powered = corr_s ** beta
```
Elevar a la potencia beta amplifica correlaciones fuertes y suprime las débiles. Esta versión **solo se usa para el baseline WGCNA**, no como entrada del GNN.

#### Paso 3 — TF Prior Boosting

```python
is_tf = np.array([g in tf_set for g in genes])  # vector booleano (N,)
tf_mask = is_tf[:, None] | is_tf[None, :]        # matriz (N, N): True si algun gen es TF

corr_p_boost = corr_p.copy()
corr_p_boost[tf_mask] *= CFG.TF_BOOST_FACTOR     # multiplica por 1.5
corr_p_boost = np.clip(corr_p_boost, 0.0, 1.0)  # mantener en [0, 1]
```

Las correlaciones que involucran al menos un TF conocido se amplifican en un 50%. Incorpora el conocimiento biológico a priori de que los TFs regulan muchos genes, sesgando la red hacia interacciones regulatorias. El clip a [0,1] mantiene la interpretación como puntuación de similitud.

#### Paso 4 — Binarización de baselines Pearson y Spearman (top-K estandarizado)

Todos los métodos se evalúan con el **mismo número de aristas** (top-K a densidad objetivo del 5%):
```python
target_edges = int(N * (N - 1) / 2 * INFERENCE_TARGET_DENSITY)  # 5% del triangulo superior
```

La función auxiliar `_topk_bin(mat)` usa `np.partition()` (O(N)) en lugar de ordenar todo el array (O(N log N)):
```python
thresh = np.partition(upper, len(upper) - target_edges)[len(upper) - target_edges]
b = (mat >= thresh).astype(float)
np.fill_diagonal(b, 0.0)
return b
```

`pearson_bin` y `spearman_bin` son las redes baseline binarizadas al mismo criterio de densidad.

#### Paso 5 — Grafo de entrada del GNN (diferente a WGCNA)

> **Decisión de diseño crítica**: El GNN recibe como grafo de entrada la red de Pearson al **percentil 75** (más denso), NO la red WGCNA al percentil 95 (más estricto). Si se usara WGCNA al 95, el GNN solo vería aristas que WGCNA ya considera fuertes: el modelo quedaría estructuralmente imposibilitado de superar a WGCNA en recall, ya que nunca podría predecir interacciones que WGCNA suprimió.

```python
input_pct = CFG.INPUT_GRAPH_PERCENTILE  # 75
input_thresh = np.percentile(corr_p_boost.flatten(), input_pct)
input_bin = (corr_p_boost >= input_thresh).astype(float)
np.fill_diagonal(input_bin, 0.0)
```

Esto produce una red más densa que WGCNA, dando al GNN acceso a más candidatos de interacción para que pueda discriminar por sí mismo cuáles son verdaderas.

#### Paso 6 — Features de nodo para el GNN

Los nodos del grafo (genes) necesitan un vector de características numérico.

**A) Features de expresión** — perfil de expresión por muestra (señal más rica):
```python
features = expr.values           # (N, num_samples) — valores brutos de expresion
norm_features = (features - mean) / (std + 1e-9)   # z-score por gen (media 0, std 1)
```
Cada fila de `norm_features` es el perfil de expresión normalizado del gen a través de todas las muestras de la condición actual.

**B) Grado de nodo** — número de conexiones en el grafo de entrada (información topológica):
```python
G_input = nx.from_numpy_array(input_bin)
degrees = np.array([d for n, d in G_input.degree()])
degrees = degrees / degrees.max()   # normalizacion [0, 1] para evitar dominancia de escala
```
El grado de un gen en el grafo de entrada captura cuántas co-expresiones tiene en la red de Pearson al 75th percentil.

**C) Centralidad de eigenvector** — importancia topológica global:
```python
# Solver ARPACK (FORTRAN compilado): ~10x mas rapido que NetworkX power-iteration
A_sp = csr_matrix(input_bin.astype(np.float32))
_, eigvecs = eigsh(A_sp, k=1, which='LM', maxiter=1000, tol=1e-3)
eig_cent = np.abs(eigvecs[:, 0])   # componente principal del espectro del grafo
```
La centralidad de eigenvector mide cuán influyente es un nodo: genes conectados a otros genes con alto grado tienen alta centralidad. Es la generalización del algoritmo PageRank de Google.

**D) Features de DESeq2** (cuando `degs_df` está disponible — enriquecimiento biológico):
```python
log2fc = degs_df.loc[g, "log2FoldChange"]    # magnitud del cambio diferencial
neg_log_padj = -np.log10(padj)              # confianza estadistica

# Normalizacion z-score para evitar dominancia de escala:
log2fc_norm       = (log2fc - log2fc.mean()) / (log2fc.std() + 1e-9)
neg_log_padj_norm = (neg_log_padj - neg_log_padj.mean()) / (neg_log_padj.std() + 1e-9)
```
`log2FoldChange` codifica la **magnitud** del cambio de expresión entre condiciones.
`-log10(padj)` codifica la **confianza estadística** (mayor valor = más significativo, ej: padj=1e-5 → neg_log_padj=5).
Añadir estos features ayuda al GNN a distinguir genes fuertemente disregulados de genes de fondo.

**Vector final de features por nodo**:
```
Sin DESeq2: [expr_norm | grado | centralidad]        -> D = num_samples + 2
Con DESeq2: [expr_norm | grado | centralidad | log2FC | -log10(padj)] -> D = num_samples + 4
```

#### Paso 7 — Construcción del objeto PyG `Data`

```python
edge_indices   = np.where(input_bin > 0)
pos_edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)  # (2, E)
pos_edge_attr  = torch.tensor(corr_p_boost[edge_indices], dtype=torch.float32).unsqueeze(1)  # (E, 1)
node_features  = torch.tensor(final_features, dtype=torch.float32)  # (N, D)

pyg_data = Data(x=node_features, edge_index=pos_edge_index, edge_attr=pos_edge_attr)
```

El objeto `Data` de PyTorch Geometric encapsula:
- `data.x`: Matriz de features de nodo `(N, D)` — `torch.float32`
- `data.edge_index`: Índices de aristas `(2, E)` — `torch.long` — cada columna `[nodo_fuente, nodo_destino]`
- `data.edge_attr`: Pesos de aristas `(E, 1)` — `torch.float32` — correlación de Pearson con TF boost
- `data.num_nodes`: N (calculado automáticamente por PyG)

---

### 4.4 Baselines Algorítmicos: ARACNE y WGCNA

#### 4.4.1 WGCNA — `build_wgcna_baseline()`

WGCNA (Weighted Gene Co-expression Network Analysis) es un método clásico de bioinformática que eleva la correlación absoluta a una potencia beta para obtener una red con propiedades aproximadamente scale-free.

```python
corr_p = |np.corrcoef(expr.values)|
np.fill_diagonal(corr_p, 0.0)
wgcna_adj = corr_p ** beta       # beta = 4: amplifica correlaciones fuertes, suprime debiles
# Binarizar al top-K (5% densidad, igual que todos los demas modelos):
wgcna_bin = (wgcna_adj >= thresh).astype(float)
```

**Salidas**: `(wgcna_bin: np.ndarray, wgcna_adj: np.ndarray)`.

#### 4.4.2 ARACNE — `build_aracne_baseline()`

ARACNE (Algorithm for the Reconstruction of Accurate Cellular Networks) usa Información Mutua (MI) y el Data Processing Inequality (DPI) para eliminar interacciones indirectas (mediadas por un gen intermediario).

**Paso 1 — Información Mutua Gaussiana**:
```python
corr_matrix = |np.corrcoef(expr.values)|
corr_matrix = np.clip(corr_matrix, 0.0, 0.9999)  # evitar log(0) cuando |rho|=1
mi_matrix = -0.5 * np.log(1.0 - corr_matrix**2)  # MI gaussiana: MI(X;Y) = -0.5*ln(1-rho^2)
```

**Paso 2 — DPI vectorizado** (el paso definitorio de ARACNE):

La DPI establece que en cualquier cadena de procesamiento de información `A → B → C`, la interacción directa más débil del triplete es probablemente indirecta.

Regla de eliminación: si `MI(i,j) < min(MI(i,k), MI(k,j)) * (1 - tolerance)` para algún intermediario `k`, la arista `(i,j)` se elimina.

Implementación vectorizada **O(N²)** en lugar de **O(N³)**:
```python
dpi = np.zeros((N, N), dtype=np.float32)
for k in range(N):
    col_k = mi_matrix[:, k:k+1]   # columna k -> shape (N, 1)
    row_k = mi_matrix[k:k+1, :]   # fila k -> shape (1, N)
    # dpi[i,j] = max_k min(MI[i,k], MI[k,j]) = fuerza de la mejor ruta alternativa via k
    np.maximum(dpi, np.minimum(col_k, row_k), out=dpi)  # in-place: sin aloc extra

tolerance = 1.0 - CFG.ARACNE_DPI_TOLERANCE  # 0.9 (10% de margen)
out_mi[(mi_matrix > 0.0) & (mi_matrix < dpi * tolerance)] = 0.0  # eliminar aristas indirectas
```

**Eficiencia**: La implementación original usaba un triple bucle Python O(N³): 4-6 horas para N=1500. La versión NumPy vectorizada tarda ~20 segundos (aceleración ~1500×).

**Paso 3 — Binarización al top-K** (mismo procedimiento que los demás métodos).

**Salidas**: `(aracne_bin: np.ndarray, out_mi: np.ndarray)`.

---

## 5. Arquitectura del Modelo GNN — `model_gnn.py`

### 5.1 TopologyAwareGATEncoder

El encoder proyecta los features de los nodos al espacio latente variacional usando capas GATv2 apiladas.

#### GATv2 vs GAT clásico

**GAT v1** (Veličković et al., 2018):
```
e_ij = LeakyReLU(a^T [W*h_i || W*h_j])
```
Problema: La transformación `W` se aplica antes de la concatenación, haciendo que `e_ij = a1*W*h_i + a2*W*h_j` sea separable en fuente y destino. Esto causa **atención estática**: los coeficientes son los mismos para cualquier query, limitando la expresividad.

**GATv2** (Brody et al., 2022):
```
e_ij = a^T LeakyReLU(W [h_i || h_j])
```
La transformación `W` se aplica DESPUÉS de la concatenación, antes de la función de activación. Los coeficientes dependen de la interacción no lineal entre fuente y destino: **atención dinámica**, más expresiva.

#### Arquitectura de capas completa

```
Entrada: (N, D) donde D = num_samples + 2 (sin DESeq2) o + 4 (con DESeq2)

[CAPA 1] GATv2Conv:
  in_channels = D
  hidden_channels = 128 (GAT_HIDDEN_CHANNELS)
  heads = 4 (GAT_HEADS)
  concat = True  -> salida: 4 * 128 = 512
  edge_dim = 1   -> usa el peso de arista en el calculo de atencion
  dropout = 0.2  -> dropout sobre coeficientes de atencion (regularizacion)
  Activacion: ReLU
  Campo receptivo: 1-hop (vecindario inmediato)
  Salida: (N, 512)

[CAPA 2] GATv2Conv:
  in_channels = 512
  hidden_channels = 128
  heads = 4
  concat = True  -> salida: 512
  edge_dim = 1, dropout = 0.2
  Activacion: ReLU
  Campo receptivo: 2-hop (vecinos de vecinos)
  Salida: (N, 512)

[CABEZA mu] GATv2Conv:
  in_channels = 512
  out_channels = 32 (GAT_OUT_CHANNELS)
  heads = 1, concat = False  -> dimension fija de salida
  edge_dim = 1, dropout = 0  (determinista, sin ruido)
  Salida: mu (N, 32) = media del espacio latente variacional

[CABEZA logstd] GATv2Conv:
  in_channels = 512
  out_channels = 32
  heads = 1, concat = False
  edge_dim = 1, dropout = 0
  Salida: logstd (N, 32) = log-desviacion estandar del espacio latente
```

**Campo receptivo**: Con 2 capas GATv2, cada gen agrega información de:
- **1-hop**: sus genes directamente co-expresados.
- **2-hop**: los genes co-expresados con sus genes co-expresados.
En términos biológicos: señal regulatoria de vecinos directos e indirectos.

**Edge attributes en GATv2**: Los pesos de arista (`edge_attr`, correlación de Pearson con TF boost) se inyectan en el cálculo de coeficientes de atención:
```
e_ij = a^T LeakyReLU(W [h_i || h_j || w_ij])
```
donde `w_ij` es el peso de la arista. Esto permite dar más atención a correlaciones fuertes durante la propagación de mensajes.

---

### 5.2 TAGAT — Autoencoder Variacional

La clase `TAGAT` envuelve el encoder dentro del marco de un Variational Graph Autoencoder (VGAE, Kipf & Welling, 2016).

#### Reparameterization trick

En un VAE, no se puede hacer backpropagation a través de una muestra aleatoria `z ~ N(mu, sigma²)` porque el operador de muestreo no es diferenciable. El reparameterization trick resuelve esto:
```
En lugar de: z ~ N(mu, sigma^2)
Se usa:      z = mu + epsilon * sigma   donde epsilon ~ N(0, I)
```
El gradiente fluye a través de `mu` y `sigma` (diferenciables), y `epsilon` es un ruido externo sin parámetros.

Implementación:
```python
def reparametrize(self, mu, logstd):
    if self.training:
        return mu + torch.randn_like(logstd) * torch.exp(logstd)
    else:
        return mu   # durante inferencia: solo la media (determinista, sin ruido)
```

El clamp `logstd.clamp(min=-10, max=2)` previene explosión de gradiente (max=2 → max sigma=7.4) y colapso del espacio latente (min=-10 → min sigma~0.00005).

#### KL Divergence Loss

```python
# KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + 2*log_sigma - mu^2 - sigma^2)
return -0.5 * torch.mean(
    torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
)
```

Penaliza distribuciones latentes que se alejan de la gaussiana estándar N(0, I). El valor de `LAMBDA_KL = 0.01` (subido de 0.0005 en la versión anterior) impone una regularización significativa del espacio latente sin suprimir la señal de reconstrucción.

---

### 5.3 Funciones de Pérdida Diferenciables — `ta_gat_loss()`

**Entradas**:
- `z: torch.Tensor (N, 32)` — representaciones latentes del encoder
- `pos_edge_index: torch.Tensor (2, E+)` — aristas positivas del grafo de entrada
- `neg_edge_index: torch.Tensor (2, E-)` — aristas negativas muestreadas aleatoriamente

#### L2-normalización (consistencia entrenamiento/inferencia)

```python
z_norm = F.normalize(z, p=2, dim=1)   # ||z_i||_2 = 1 para todo i
```
Fuerza que el producto interno entre dos vectores L2-normalizados sea la **similitud coseno**:
`z_norm[i] · z_norm[j] = cos(theta_ij)`.

Esto es **crítico para la consistencia** entre entrenamiento e inferencia: ambas etapas usan la misma métrica (coseno normalizado). En versiones anteriores, el entrenamiento usaba producto interno crudo y la inferencia usaba coseno, produciendo scores incompatibles.

#### Pérdida de reconstrucción (Binary Cross-Entropy)

```python
pos_pred = sigmoid(sum(z_norm[pos_edge_index[0]] * z_norm[pos_edge_index[1]], dim=1))
neg_pred = sigmoid(sum(z_norm[neg_edge_index[0]] * z_norm[neg_edge_index[1]], dim=1))

pos_loss   = -log(pos_pred + EPS).mean()     # maximizar prob. de aristas reales
neg_loss   = -log(1 - neg_pred + EPS).mean() # minimizar prob. de aristas falsas
recon_loss = pos_loss + neg_loss
```

El **negative sampling** es esencial: como el grafo es disperso (~5-10% de pares posibles), si solo se entrenara con positivos, el modelo aprendería trivialmente a predecir 1 para todo. Las aristas negativas son pares de genes que **no** tienen arista en el grafo de entrada.

Con `DYNAMIC_NEG_SAMPLING=True`, las aristas negativas se re-muestrean en cada época, mejorando la generalización al explorar continuamente el espacio completo de no-interacciones.

#### Grado suave (sparse, O(E) en lugar de O(N²))

```python
soft_degrees = torch.zeros(N, device=z.device)
soft_degrees.scatter_add_(0, pos_edge_index[0], pos_pred)  # acumular prob. en nodo fuente
soft_degrees.scatter_add_(0, pos_edge_index[1], pos_pred)  # acumular prob. en nodo destino
```
`soft_degrees[i]` = suma de probabilidades predichas de todas las aristas que inciden en el nodo i. Es la versión diferenciable del grado del nodo, computada sparely sobre las aristas de entrenamiento en lugar de construir la matriz NxN completa.

#### Pérdida Scale-Free: `scale_free_loss_v2(degrees, gamma=2.5)`

**Motivación biológica**: Las redes de regulación génica reales siguen una distribución de grado de ley de potencia `P(k) ~ k^-gamma` (Barabási-Albert, 1999): la mayoría de genes tienen pocos vecinos y unos pocos genes tienen muchos (hubs, frecuentemente TFs). Forzar esta propiedad durante el entrenamiento orienta el modelo hacia redes biológicamente plausibles.

**Implementación — regresión lineal diferenciable en log-log**:
1. Ordena los nodos por grado soft descendiente y asigna rangos 1, 2, ..., N.
2. En escala log-log, una distribución scale-free muestra `log(rango) ~ -gamma * log(grado)` → línea recta con pendiente negativa.
3. Calcula R² del ajuste lineal de forma diferenciable via PyTorch.

Tres términos de pérdida:
- **Linealidad** `(1 - R²)`: Minimizar → la distribución en log-log se acerca a una recta.
- **Pendiente** `relu(slope + 0.3)`: Penaliza pendientes positivas (distribución no decreciente = no scale-free).
- **Uniformidad** `1 / (coeff_of_variation + 0.1)`: Penaliza redes donde todos los genes tienen el mismo grado (queremos hubs, no red regular).

```python
loss = linearity_loss + 0.5 * slope_penalty + 0.05 * uniformity_penalty
```

#### Pérdida de Esparcidad: `targeted_sparsity_loss(degrees)`

```python
norm_degrees = degrees / N   # normalizar por N para independencia del tamaño de la red
min_degree_penalty    = torch.min(norm_degrees)    # penalizar si el minimo es alto
median_degree_penalty = sorted_norm_deg[N // 2]   # penalizar si la mediana es alta

loss = 0.5 * min_degree_penalty + 0.5 * median_degree_penalty
```

Esta pérdida **permite** que algunos nodos (hubs) tengan muchas conexiones, pero fuerza a que la mayoría de genes (mediana y mínimo) tengan pocos vecinos. Resultado: red dispersa con hubs = estructura biológicamente plausible.

#### Pérdida total ponderada

```
Loss_total = 5.0  × recon_loss    (BCE: señal principal de aprendizaje)
           + 0.01 × kl_div        (KL: regularizacion espacio latente)
           + 2.0  × sf_loss       (scale-free: topologia biologica)
           + 0.3  × sparsity_loss (sparsity: red dispersa con hubs)
```

---

## 6. Entrenamiento del Modelo — `main.py`

### 6.1 Ciclo de Entrenamiento y Early Stopping — `train_ta_gat()`

```python
model.train()    # activa dropout, reparametrizacion estocastica

for epoch in range(epochs):   # max 500 epocas
    optimizer.zero_grad()     # limpiar gradientes acumulados

    # 1. Muestreo de aristas negativas (re-muestreo dinamico si DYNAMIC_NEG_SAMPLING=True)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1),  # igual numero que aristas positivas
        method='sparse'          # metodo eficiente para grafos dispersos
    ).to(torch.long).to(device)

    # 2. Forward pass del encoder
    z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
    # Durante training: z = mu + epsilon*sigma   (con ruido, shape N x 32)

    # 3. Calculo de todos los terminos de perdida
    recon_loss, sf_loss, sparsity_loss = ta_gat_loss(
        z, data.edge_index, neg_edge_index,
        CFG.LAMBDA_SCALE_FREE, pos_edge_weights=data.edge_attr
    )
    kl_div = (1.0 / data.num_nodes) * model.kl_loss()  # KL normalizado por N

    # 4. Perdida total ponderada
    loss = (CFG.LAMBDA_RECON    * recon_loss
          + CFG.LAMBDA_KL       * kl_div
          + CFG.LAMBDA_SCALE_FREE * sf_loss
          + CFG.LAMBDA_SPARSITY * sparsity_loss)

    # 5. Backpropagation: calcula gradientes de todos los parametros del modelo
    loss.backward()

    # 6. Actualizacion de parametros con Adam
    optimizer.step()

    # 7. Early stopping: detener si la loss no mejora lo suficiente
    if current_loss < best_loss - CFG.MIN_DELTA:
        best_loss = current_loss
        patience_counter = 0       # resetear contador
    else:
        patience_counter += 1      # incrementar contador

    if patience_counter >= CFG.PATIENCE:  # 50 epocas consecutivas sin mejora
        break
```

#### Optimizador Adam

```python
optimizer = optim.Adam(model.parameters(), lr=CFG.LR)  # LR = 0.001
```

Adam (Adaptive Moment Estimation, Kingma & Ba, 2015) ajusta automáticamente la tasa de aprendizaje por parámetro usando estimaciones del primer y segundo momento del gradiente. Es más eficiente que SGD para optimización no convexa como el entrenamiento de GNNs.

#### Logging durante entrenamiento

Cada 50 épocas y en la última:
```
Epoch 050/500 | Loss: 3.2451 [Rec: 2.1234, KL: 0.0123, SF: 0.8765, Sparsity: 0.0329]
```

Si el dispositivo es Apple MPS, se llama `torch.mps.empty_cache()` cada 50 épocas para liberar memoria de la GPU.

---

### 6.2 Pipeline Multi-Seed — `run_multi_seed_pipeline()`

El multi-seed es fundamental para la **validez estadística** de los resultados del paper. Los GNNs tienen:
- Inicialización aleatoria de pesos (glorot uniform).
- Reparametrización estocástica (`epsilon ~ N(0, I)` en cada paso).
- Negative sampling aleatorio en cada época.

Reportar un único resultado puede estar sesgado hacia esa semilla específica. El estándar en publicaciones GRN es reportar media ± desviación estándar sobre ≥ 5 semillas.

```python
seeds = [123] if FAST_MODE else CFG.N_SEEDS  # [42, 123, 456] por defecto

all_results: list[dict] = []
all_scores:  list[np.ndarray] = []

for seed in seeds:
    set_seed(seed)   # reproducibilidad: fija todas las semillas aleatorias
    seed_results, seed_adj, seed_score = run_condition_pipeline(...)
    all_results.append(seed_results)   # metricas de rendimiento
    all_scores.append(seed_score)      # matriz de puntuaciones NxN para ensemble

averaged = average_multi_seed_results(all_results)
```

#### Red de consenso ensemble

En lugar de elegir la red de la "mejor semilla" (que refleja solo una inicialización aleatoria), se promedian las matrices de puntuación continua de todas las semillas:
```python
avg_score = np.mean(all_scores, axis=0)  # promedio elemento a elemento -> (N, N)
# Binarizar el promedio a la densidad objetivo:
ensemble_adj = (avg_score >= thr_val).astype(float)
```
La red ensemble es más estable que cualquier red individual porque el promedio suaviza el ruido estocástico de la inicialización.

#### Promediado de métricas: `average_multi_seed_results()`

Para cada gold standard y modelo, calcula media ± std de todas las métricas numéricas:
```python
for key in numeric_keys:
    vals = [r[gs_name][model_name][key] for r in all_results_list]
    metrics_agg[key]          = float(np.mean(vals))   # media
    metrics_agg[key + "_std"] = float(np.std(vals))    # desviacion estandar
```

Las métricas de metadatos (`Pred_Edges`, `True_Edges`, `Common_Genes`) se toman de la primera semilla (son deterministas dado el mismo grafo de entrada).

#### Test de Wilcoxon signed-rank — `wilcoxon_pairwise_test()`

Si hay ≥ 2 semillas, para cada baseline `b` en {Pearson, Spearman, ARACNE, WGCNA}:
```python
# H0: TA-GAT no supera al baseline
# H1 (alternative='greater'): TA-GAT > baseline
stat, p = scipy.stats.wilcoxon(tagat_vals, baseline_vals, alternative='greater')
```

Convenciones de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001.

> **Nota importante**: El test de Wilcoxon requiere ≥ 5-6 semillas para tener potencia estadística razonable. Con 3 semillas (N_SEEDS por defecto) el test es orientativo. Para el paper final se recomienda ≥ 5 semillas.

---

## 7. Inferencia de Red — `infer_network()`

Una vez entrenado el modelo, se usa el encoder para generar representaciones latentes y derivar la red inferida.

```python
model.eval()   # desactiva dropout y usa z = mu (determinista, sin ruido)

with torch.no_grad():   # deshabilita autograd para ahorrar memoria
    z = model.encode(data.x, data.edge_index, edge_attr=edge_attr)
    # Durante eval: reparametrize devuelve mu directamente (shape N x 32)

    z_norm = F.normalize(z, p=2, dim=1)   # L2-normalizacion (misma que en training)
    adj_pred = torch.sigmoid(torch.matmul(z_norm, z_norm.t()))
    # adj_pred[i,j] = sigmoid(cos_similarity(z_i, z_j)) in [0, 1]
    # Interpretacion: probabilidad de interaccion entre gen i y gen j
    # Complejidad: O(N^2) = unico punto O(N^2) en todo el pipeline (inevitable en inferencia)
```

La multiplicación matricial `z_norm @ z_norm.T` calcula simultáneamente todas las N² similitudes coseno. El sigmoid mapea [-1, 1] → [0, 1] como probabilidad de interacción.

**Binarización top-K** (consistente con todos los baselines):
```python
N = score_np.shape[0]
target_edges = int(N * (N-1) / 2 * INFERENCE_TARGET_DENSITY)  # 5% del triangulo superior
upper_scores = score_np[np.triu_indices_from(score_np, k=1)]  # triangulo superior
# np.partition() selecciona el umbral en O(N) sin ordenar el array completo:
threshold_val = np.partition(upper_scores, len(upper_scores) - target_edges)[len(upper_scores) - target_edges]
adj_bin = (score_np >= threshold_val).astype(float)
np.fill_diagonal(adj_bin, 0.0)
adj_bin = np.maximum(adj_bin, adj_bin.T)   # garantizar simetria exacta
```

**Salidas**:
- `adj_bin: np.ndarray (N×N)` — Red binaria inferida (utilizada para métricas de clasificación).
- `score_np: np.ndarray (N×N)` — Puntuaciones continuas (utilizadas para AUC, AUPRC, P@K, curvas ROC/PR).

---

## 8. Evaluación y Métricas — `metrics.py`

### 8.1 Evaluación Justa (Fair Evaluation)

La evaluación se restringe al subconjunto de genes que aparecen tanto en la lista de DEGs del experimento como en el Gold Standard. Genes que están en los DEGs pero no en el GS no pueden evaluarse correctamente (no sabemos si son verdaderos positivos o negativos). Esta restricción se llama "Fair Evaluation" y es el estándar en benchmarks GRN modernos.

```python
eval_indices = [i for i, g in enumerate(gene_names) if g in evaluable_genes_set]
sub_pred  = pred_adj[np.ix_(eval_indices, eval_indices)]   # sub-red solo con genes evaluables
sub_truth = truth_adj[np.ix_(eval_indices, eval_indices)]
# Solo triangulo superior (aristas unicas en red no dirigida simetrica):
upper_mask = np.triu(np.ones_like(sub_truth, dtype=bool), k=1)
y_pred = sub_pred[upper_mask].astype(int)  # vector de predicciones binarias
y_true = sub_truth[upper_mask].astype(int)  # vector de verdad binaria
```

### 8.2 Métricas de Clasificación Binaria — `evaluate_against_gold_standard()`

Calculadas con `sklearn.metrics` sobre los vectores `y_pred` y `y_true` del subgrafo evaluable:

| Métrica | Fórmula | Interpretación biológica |
|---|---|---|
| **Precision** | TP / (TP + FP) | De las aristas predichas, ¿cuántas son interacciones reales? |
| **Recall** | TP / (TP + FN) | De las interacciones reales, ¿cuántas fueron detectadas? |
| **F1-Score** | 2×P×R / (P+R) | Balance entre Precision y Recall |
| **Accuracy** | (TP+TN) / total | Tasa de clasificación correcta global |
| **Pred_Edges** | sum(y_pred) | Número de aristas predichas en la sub-red |
| **True_Edges** | sum(y_true) | Número de aristas reales en la sub-red |
| **Common_Genes** | len(eval_indices) | Número de genes evaluables |

### 8.3 AUC-ROC — `compute_auc_against_gs()`

Evalúa la capacidad discriminativa del modelo usando las puntuaciones continuas (no binarizadas):
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_score)
```
0.5 = clasificador aleatorio; 1.0 = perfecto.

**Limitación**: En redes muy dispersas (5% de densidad positiva), el AUC-ROC puede ser optimistamente alto porque hay muchos verdaderos negativos que se clasifican correctamente con cualquier threshold. Por eso el AUPRC es más informativo para este problema.

### 8.4 AUPRC — `compute_auprc_against_gs()`

Métrica preferida para clasificación desbalanceada (mucho más informativa que AUC en redes GRN dispersas):
```python
from sklearn.metrics import average_precision_score
auprc = average_precision_score(y_true, y_score)
```
Un clasificador aleatorio con 5% de positivos tiene AUPRC ≈ 0.05. Cualquier valor significativamente superior indica poder predictivo real. La mayoría de benchmarks GRN modernos (GENIE3, SCENIC+, VIPER) reportan Average Precision.

### 8.5 Precision@K — `compute_precision_at_k()`

¿Qué fracción de las K aristas con mayor puntuación son verdaderos positivos?
```python
sorted_idx = np.argsort(y_score)[::-1]   # ordenar por puntuacion descendente
for k in [50, 100, 200, 500]:
    effective_k = min(k, len(sorted_idx))
    P_at_K = y_true[sorted_idx[:effective_k]].mean()
```

P@K es directamente interpretable en términos experimentales: si P@100 = 0.3, las 100 predicciones más confiadas del modelo son interacciones reales en un 30%, cuando el azar esperado daría solo el 5%.

### 8.6 Métricas Topológicas — `compute_network_topology_metrics()`

| Métrica | Descripción |
|---|---|
| `Nodes` | Número total de nodos (genes) |
| `Edges` | Número total de aristas (interacciones) |
| `Avg Degree` | Grado medio: `2×E / N` |
| `Clustering Coef` | Coeficiente de clustering promedio: densidad de triángulos locales |
| `Diameter (LCC)` | Diámetro del Largest Connected Component (estimado con double-sweep BFS) |
| `Density` | Densidad global: `2×E / (N×(N-1))` |
| `Scale-Free R²` | R² del ajuste de la distribución de grados a ley de potencia en escala log-log |

#### `compute_scale_free_fit(G)` — Ajuste ley de potencia

```python
degree_counts = nx.degree_histogram(G)  # frecuencia de cada grado posible
for degree, count in enumerate(degree_counts):
    if count > 0 and degree > 0:
        x.append(np.log10(degree))       # log10(k)
        y.append(np.log10(count / N))    # log10(P(k))
slope, intercept, r_value, p_value, std_err = linregress(x, y)
return r_value ** 2   # R^2 del ajuste lineal en log-log
```

R² cercano a 1 indica que la red tiene distribución de grados de ley de potencia (scale-free). Las redes de regulación génica reales suelen tener R² > 0.7. Las redes aleatorias (Erdős-Rényi) tienen R² cercano a 0.

#### `_approx_diameter(G_sub)` — Diámetro por double-sweep BFS

El diámetro exacto requiere N BFS completos O(N×(N+E)). Para N=1500 y redes densas puede tardar minutos. El algoritmo double-sweep es O(N+E):
1. BFS desde un nodo arbitrario → encuentra el nodo `far1` más lejano.
2. BFS desde `far1` → encuentra el nodo `far2` más lejano de `far1`.
3. La distancia `far1 → far2` es una cota inferior ajustada del diámetro real (exacta en árboles y muy precisa en redes reales).

### 8.7 Gráficas ROC y Precision-Recall — `plot_roc_curves()`, `plot_pr_curves()`

Se generan 6 figuras por condición experimental (3 Gold Standards × 2 tipos de curva):
- Curvas ROC: FPR vs TPR con AUC para cada modelo.
- Curvas PR: Recall vs Precision con Average Precision para cada modelo.

Las curvas PR incluyen una línea de referencia del clasificador aleatorio (línea horizontal al ratio de positivos, ej. 5%) para contextualizar las mejoras. Guardadas como PNG a 300 DPI.

---

## 9. Análisis de Relevancia Biológica — `biological_relevance.py`

### 9.1 Análisis de Hubs — `analyze_hubs()`

Identifica los genes más conectados (hubs) en la red inferida y determina qué porcentaje son factores de transcripción conocidos.

```python
degrees = np.sum(adj, axis=1)   # grado de cada gen = suma de su fila en la matriz binaria
df_genes = pd.DataFrame({'Gene': genes, 'Degree': degrees, 'is_TF': [g in tf_set for g in genes]})
top_20 = df_genes.sort_values('Degree', ascending=False).head(20)
```

**Salidas**:
- `top_20_hubs_{condition}.csv` por condición con Gene, Degree, is_TF.
- `hubs_comparison.csv`: tabla resumen con `[Condición, Total_Hubs, Num_TFs, PCT_TFs]`.

**Interpretación biológica**: En redes GRN reales, los hubs suelen ser factores de transcripción (TP53, MYC, EGFR, etc.). Un alto porcentaje de TFs entre los top 20 hubs es evidencia de que la red captura estructura regulatoria real, validando la calidad de la inferencia sin necesidad del Gold Standard.

### 9.2 Red Diferencial — `compute_differential_network()`

Identifica interacciones que se **ganan** o **pierden** en la transición biológica (ej. Normal → Tumor).

```python
gained_in_a = np.clip(adj_a - adj_b, 0, 1)  # aristas en Tumor pero no en Normal
lost_in_a   = np.clip(adj_b - adj_a, 0, 1)  # aristas en Normal pero no en Tumor
net_diff = gained_in_a.sum(axis=1) - lost_in_a.sum(axis=1)  # +: mas conexiones en Tumor
```

**Salidas**: `differential_network_{A}_vs_{B}.csv` con columnas: Gene, Degree_Tumor, Degree_Normal, Gained_in_A, Lost_in_A, Net_Differential.

Los genes con mayor `|Net_Differential|` son los más "rewired" en la transición biológica: han ganado o perdido muchas interacciones específicamente en una condición. Estos son candidatos experimentales prioritarios para validación.

---

## 10. Salidas del Sistema

Toda la salida se organiza en `new_implementation/out/{dataset_id}/`:

```
out/{dataset_id}/
|
+-- run.log                               <- log completo de toda la ejecucion
|                                            (stdout + stderr, via clase Logger)
+-- top_20_hubs_tumor.csv                 <- top 20 hubs de la condicion Tumor
+-- top_20_hubs_normal.csv                <- top 20 hubs de la condicion Normal
+-- hubs_comparison.csv                   <- resumen comparativo de hubs
+-- differential_network_Tumor_vs_Normal.csv <- red diferencial
|
+-- tumor/                                <- outputs de la condicion Tumor
|   +-- pearson_network.tsv               <- red Pearson binaria (N x N)
|   +-- spearman_network.tsv              <- red Spearman binaria (N x N)
|   +-- aracne_network.tsv                <- red ARACNE binaria (N x N)
|   +-- wgcna_network.tsv                 <- red WGCNA binaria (N x N)
|   +-- tagat_network.tsv                 <- red TA-GAT ensemble binaria (N x N)
|   +-- figures/
|       +-- roc_{dataset}_tumor_biogrid.png   <- curva ROC vs BioGRID
|       +-- roc_{dataset}_tumor_string.png    <- curva ROC vs STRING
|       +-- roc_{dataset}_tumor_genemania.png <- curva ROC vs GeneMANIA
|       +-- pr_{dataset}_tumor_biogrid.png    <- curva PR vs BioGRID
|       +-- pr_{dataset}_tumor_string.png     <- curva PR vs STRING
|       +-- pr_{dataset}_tumor_genemania.png  <- curva PR vs GeneMANIA
|
+-- normal/                               <- outputs de la condicion Normal (idem)
|
+-- figures/
    +-- learning_curve_tumor.png          <- curva de aprendizaje del GNN (Tumor)
    +-- learning_curve_normal.png         <- curva de aprendizaje del GNN (Normal)
```

**Nota sobre logging**: `sys.stdout` y `sys.stderr` son redirigidos por el objeto `Logger` a `run.log`. Todo lo que se imprime en consola queda persistido automáticamente en el archivo de log.

---

## 11. Flujo Completo — Diagrama Paso a Paso

```
INICIO: python main.py
|
+-- 1. run_interactive_setup()
|     -> Determina modo de carga de datos (local CSV o GEO)
|     -> Configura CFG.GEO_ID, CFG.GEO_GROUPS, CFG.GEO_CONTROL_GROUP, etc.
|     -> Crea directorios de salida out/{dataset_id}/ y figures/
|     -> Devuelve (out_dir, figures_dir)
|
+-- 2. Logger(out_dir/"run.log")
|     -> Redirige sys.stdout y sys.stderr al archivo de log
|
+-- 3. set_seed(CFG.SEED = 123)
|     -> random.seed(123), np.random.seed(123), torch.manual_seed(123)
|     -> cuda.manual_seed_all(123) o mps.manual_seed(123)
|     -> torch.backends.cudnn.deterministic = True
|     -> os.environ['PYTHONHASHSEED'] = '123'
|
+-- 4. Deteccion de hardware
|     CUDA disponible  -> device = torch.device('cuda')
|     MPS disponible   -> device = torch.device('mps')  (Apple Silicon)
|     CPU              -> device = torch.device('cpu')
|
+-- 5. load_tf_list(CFG.TF_LIST_PATH)
|     -> Lee data/TF_names_v_1.01.txt (un TF por linea)
|     -> Devuelve tf_set: set[str] (~1800 TFs humanos conocidos)
|
+-- 6. Carga de expresion + DESeq2
|     Si LOCAL_MODE=True:
|       load_local_counts_and_run_deseq2()
|       Lee counts_control_CLEAN.csv + counts_primary_CLEAN.csv
|     Si LOCAL_MODE=False:
|       load_geo_and_run_deseq2()
|       Descarga GSE{id} via GEOparse
|     Ambos ejecutan PyDESeq2 con el mismo protocolo
|     Salidas:
|       sig_degs: pd.DataFrame (log2FC, padj, ...) indexado por gen
|       expr_groups_dict: dict{grupo: DataFrame(genes x muestras)}
|       degs_genes: list[str] (max 1500 DEGs ordenados por padj)
|
+-- 7. Gold Standards (3 bases de datos, en paralelo conceptualmente):
|
|     download_and_parse_biogrid()
|       Si no existe data/BIOGRID-*.tab3.txt: descarga ZIP ~40MB, extrae
|       Lee el archivo, filtra TaxID=9606, deduplica, normaliza
|       -> unique_bg_edges: set[tuple[str, str]]  (~600K-700K aristas)
|
|     download_and_parse_string()
|       Si no existen los .gz: descarga protein.links + protein.info
|       Construye dict ENSP->gene, filtra score>=400, deduplica
|       -> unique_string_edges: set[tuple[str, str]]
|
|     download_and_parse_genemania_coexp()
|       Descarga id_mappings.txt + 3 archivos de co-expresion
|       Traduce a simbolos HUGO, deduplica
|       -> unique_gm_edges: set[tuple[str, str]]
|
|     create_ground_truth_adj(degs_genes, unique_bg_edges)   -> (bg_adj NxN, bg_eval_set)
|     create_ground_truth_adj(degs_genes, unique_string_edges) -> (str_adj, str_eval_set)
|     create_ground_truth_adj(degs_genes, unique_gm_edges)   -> (gm_adj, gm_eval_set)
|
+-- 8. Bucle por condicion experimental:
|     (ej: para g_name='Tumor', g_expr=DataFrame(1500 genes x 50 muestras))
|
|     run_multi_seed_pipeline(
|         condition_name='Tumor', dataset_id, g_expr,
|         bg_adj, bg_eval_set, str_adj, str_eval_set, gm_adj, gm_eval_set,
|         degs_genes, tf_set, device, sig_degs=sig_degs
|     )
|     |
|     +-- seeds = [42, 123, 456]  (o [123] si FAST_MODE)
|     |
|     +-- BUCLE for seed in seeds:
|     |   |
|     |   +-- set_seed(seed)
|     |   |
|     |   +-- run_condition_pipeline(condition_name='Tumor', ...)
|     |   |   |
|     |   |   +-- A. build_correlation_baseline(expr=g_expr, tf_set, degs_df=sig_degs)
|     |   |   |   |
|     |   |   |   +-- Pearson y Spearman absolutas (N x N)
|     |   |   |   +-- WGCNA powered (N x N) [solo para baseline]
|     |   |   |   +-- TF Prior Boosting: corr_p_boost[tf_mask] *= 1.5
|     |   |   |   +-- top-K binarizacion: pearson_bin, spearman_bin (5% densidad)
|     |   |   |   +-- GNN input graph: input_bin = Pearson al 75th pct
|     |   |   |   +-- Node features:
|     |   |   |   |   [z-score expresion | grado/max | centralidad eigenvec]
|     |   |   |   |   + [log2FC_norm | -log10(padj)_norm]  (si degs_df disponible)
|     |   |   |   +-- pyg_data = Data(x=(N,D), edge_index=(2,E), edge_attr=(E,1))
|     |   |   |   |
|     |   |   |   Devuelve:
|     |   |   |   input_bin, pearson_bin, spearman_bin,
|     |   |   |   corr_p_boost, pearson_score, spearman_score, pyg_data
|     |   |   |
|     |   |   +-- B. pyg_data = pyg_data.to(device)  [mover al hardware]
|     |   |   |
|     |   |   +-- C. encoder = TopologyAwareGATEncoder(
|     |   |   |         in_channels=D, hidden_channels=128, out_channels=32, heads=4,
|     |   |   |         dropout=0.2
|     |   |   |     )
|     |   |   |     model = TAGAT(encoder).to(device)
|     |   |   |     optimizer = optim.Adam(model.parameters(), lr=0.001)
|     |   |   |
|     |   |   +-- D. train_ta_gat(model, pyg_data, optimizer, epochs=500)
|     |   |   |   |
|     |   |   |   +-- Para cada epoca (max 500):
|     |   |   |   |   optimizer.zero_grad()
|     |   |   |   |   neg_edge_index = negative_sampling(...)  [dinamico]
|     |   |   |   |   z = model.encode(x, edge_idx, edge_attr) [con reparametrizacion]
|     |   |   |   |   recon, sf, spar = ta_gat_loss(z, pos_edges, neg_edges)
|     |   |   |   |   kl = model.kl_loss() / N
|     |   |   |   |   loss = 5*recon + 0.01*kl + 2*sf + 0.3*spar
|     |   |   |   |   loss.backward()
|     |   |   |   |   optimizer.step()
|     |   |   |   |   [early stopping: PATIENCE=50, MIN_DELTA=1e-4]
|     |   |   |   |
|     |   |   |   Devuelve loss_history: list[float]
|     |   |   |
|     |   |   +-- E. plot_learning_curve(loss_history, 'Tumor') -> PNG
|     |   |   |
|     |   |   +-- F. infer_network(model, pyg_data)
|     |   |   |   |
|     |   |   |   +-- model.eval()  [z = mu, sin reparametrizacion]
|     |   |   |   +-- z_norm = F.normalize(model.encode(...), p=2, dim=1)
|     |   |   |   +-- score = sigmoid(z_norm @ z_norm.T)  -> (N, N) en [0, 1]
|     |   |   |   +-- top-K binarizacion al 5% densidad
|     |   |   |   +-- simetrizar: adj_bin = max(adj_bin, adj_bin.T)
|     |   |   |   |
|     |   |   |   Devuelve ta_gat_bin (NxN binaria), tagat_score (NxN continua)
|     |   |   |
|     |   |   +-- G. build_aracne_baseline(expr=g_expr)
|     |   |   |     MI gaussiana -> DPI vectorizado O(N^2) -> top-K bin
|     |   |   |     Devuelve (aracne_bin, aracne_score)
|     |   |   |
|     |   |   +-- H. build_wgcna_baseline(expr=g_expr)
|     |   |   |     Pearson^beta -> top-K bin
|     |   |   |     Devuelve (wgcna_bin, wgcna_score)
|     |   |   |
|     |   |   +-- I. models_dict = {
|     |   |   |       'Pearson': pearson_bin,
|     |   |   |       'Spearman': spearman_bin,
|     |   |   |       'ARACNE': aracne_bin,
|     |   |   |       'WGCNA': wgcna_bin,
|     |   |   |       'TA-GAT': ta_gat_bin
|     |   |   |     }
|     |   |   |     scores_dict = {
|     |   |   |       'Pearson': pearson_score,
|     |   |   |       'Spearman': spearman_score,
|     |   |   |       'ARACNE': aracne_score,
|     |   |   |       'WGCNA': wgcna_score,
|     |   |   |       'TA-GAT': tagat_score
|     |   |   |     }
|     |   |   |
|     |   |   +-- J. evaluate_against_gold_standard() x 5 modelos x 3 GS
|     |   |   |     -> Precision, Recall, F1, Accuracy en sub-red evaluable
|     |   |   |
|     |   |   +-- K. compute_auc_against_gs()   x 5 modelos x 3 GS -> AUC-ROC
|     |   |       compute_auprc_against_gs()  x 5 modelos x 3 GS -> AUPRC
|     |   |       compute_precision_at_k()    x 5 modelos x 3 GS -> P@{50,100,200,500}
|     |   |   |
|     |   |   +-- L. compute_network_topology_metrics() x 5 modelos
|     |   |   |     -> Nodes, Edges, Avg Degree, Clustering, Diameter, SF-R2, Density
|     |   |   |
|     |   |   +-- M. Guardar redes TSV en out/{dataset}/tumor/
|     |   |   |     pearson_network.tsv, spearman_network.tsv, aracne_network.tsv,
|     |   |   |     wgcna_network.tsv, tagat_network.tsv
|     |   |   |
|     |   |   +-- N. Guardar figuras en out/{dataset}/tumor/figures/
|     |   |         6 curvas ROC + 6 curvas PR (3 GS x 2 tipos)
|     |   |
|     |   +-- Devuelve (results_dict, ta_gat_bin, tagat_score)
|     |
|     +-- Fin del bucle de semillas
|     |
|     +-- average_multi_seed_results(all_results)
|     |     -> Para cada GS x modelo: media y std de todas las metricas
|     |
|     +-- Ensemble network:
|     |     avg_score = np.mean(all_scores, axis=0)  [promedio de score matrices]
|     |     ensemble_adj = binarizar avg_score al 5% densidad
|     |
|     +-- wilcoxon_pairwise_test() (si >= 2 semillas)
|     |     Prueba unilateral TA-GAT > {Pearson, Spearman, ARACNE, WGCNA}
|     |     Reporta: estadistico, p-valor, stars de significancia
|     |
|     +-- Devuelve (averaged_results_dict, ensemble_adj)
|
|     all_final_results['Tumor'] = averaged
|     adj_dict['Tumor'] = ensemble_adj
|
|     [Mismo bucle para 'Normal' si existe]
|
+-- print_final_summary_tables(all_final_results)
|     Tabla 1: AUC, AUPRC, P@100, Precision, Recall, F1 por condicion y GS
|     Tabla 2: Topologia (Nodes, Edges, Avg Degree, Clustering, Diameter, SF-R2)
|
+-- analyze_hubs(adj_dict, degs_genes, tf_set, config.OUT_DIR)
|     Para cada condicion: top 20 genes por grado, porcentaje de TFs
|     Guarda top_20_hubs_{cond}.csv y hubs_comparison.csv
|
+-- compute_differential_network(adj_dict, degs_genes, config.OUT_DIR)
|     Ganancias y perdidas de aristas entre condiciones
|     Guarda differential_network_Tumor_vs_Normal.csv

FIN: "Todas las ejecuciones de TA-GAT han terminado con exito."
```

---

## Notas Finales para el Paper

### Contribuciones técnicas principales

1. **Arquitectura VAE + GATv2 de 2 capas** con campo receptivo de 2-hop y dropout de atención para datasets pequeños de DEGs.
2. **Función de pérdida multi-objetivo** que combina BCE de reconstrucción, KL variacional, regularización scale-free diferenciable y pérdida de esparcidad dirigida — todo O(E) en lugar de O(N²).
3. **TF Prior Boosting**: incorporación del conocimiento biológico de factores de transcripción como peso a priori sobre las aristas del grafo de entrada.
4. **Separación grafo de entrada / grafo de evaluación**: el GNN recibe Pearson al 75th percentil (más denso), todos los modelos se evalúan al mismo top-5% de densidad (comparación justa).
5. **Features de nodo enriquecidos con DESeq2**: log2FC y -log10(padj) como señales de diferenciación génica directamente en el input del GNN.
6. **DPI vectorizado de ARACNE**: de O(N³) Python a O(N²) NumPy (~1500× más rápido), haciendo ARACNE viable como baseline competitivo.
7. **Evaluación justa multi-métrica**: AUC-ROC, AUPRC, P@K y F1 restringidos al subconjunto evaluable de genes, con promediado multi-semilla y tests de Wilcoxon.

### Limitaciones a mencionar en el paper

- El cap de 1500 DEGs limita el análisis a redes de tamaño manejable. Datasets con muy pocos DEGs (<50) pueden producir redes poco representativas.
- El modelo asume interacciones no dirigidas (red simétrica). Las interacciones de regulación son frecuentemente dirigidas (TF → gen diana), y esta información se pierde.
- El test de Wilcoxon requiere ≥ 5-6 semillas para tener potencia estadística real; con 3 semillas (N_SEEDS por defecto) el resultado es orientativo.
- El TF Prior Boosting puede introducir sesgo hacia genes previamente conocidos como reguladores, potencialmente penalizando la detección de TFs no caracterizados.
- La similitud coseno como métrica de reconstrucción asume que la similitud entre embeddings latentes es una buena proxy de la interacción biológica, lo cual puede no ser siempre cierto.
