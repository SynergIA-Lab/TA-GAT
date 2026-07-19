# Documentación Completa del Algoritmo TA-GAT

> **¿Para qué sirve este archivo?**
> Este documento explica en lenguaje normal todo lo que hace el algoritmo TA-GAT: qué problema resuelve, cómo está organizado el código, qué hace cada archivo y cada función, cómo se procesan los datos antes y después de la red neuronal, cómo funciona la red en sí, y qué parámetros controlan todo.
> Se actualiza cada vez que hay un cambio relevante en el código.

---

## Índice

1. [¿Qué problema resuelve TA-GAT?](#1-qué-problema-resuelve-ta-gat)
2. [Visión general del pipeline](#2-visión-general-del-pipeline)
3. [Arquitectura de ficheros](#3-arquitectura-de-ficheros)
4. [PASO 0 — Configuración inicial](#4-paso-0--configuración-inicial)
5. [PASO 1 — Carga y preprocesamiento de datos](#5-paso-1--carga-y-preprocesamiento-de-datos)
6. [PASO 2 — Construcción del grafo de entrada para la GNN](#6-paso-2--construcción-del-grafo-de-entrada-para-la-gnn)
7. [PASO 3 — La Red Neuronal: arquitectura TA-GAT](#7-paso-3--la-red-neuronal-arquitectura-ta-gat)
8. [PASO 4 — Entrenamiento de la GNN](#8-paso-4--entrenamiento-de-la-gnn)
9. [PASO 5 — Inferencia de la red final](#9-paso-5--inferencia-de-la-red-final)
10. [PASO 6 — Baselines de comparación](#10-paso-6--baselines-de-comparación)
11. [PASO 7 — Evaluación y métricas](#11-paso-7--evaluación-y-métricas)
12. [PASO 8 — Análisis biológico post-inferencia](#12-paso-8--análisis-biológico-post-inferencia)
13. [Parámetros del modelo: tabla de referencia](#13-parámetros-del-modelo-tabla-de-referencia)
14. [Flujo de ejecución completo (resumen)](#14-flujo-de-ejecución-completo-resumen)
15. [Historial de cambios relevantes](#15-historial-de-cambios-relevantes)

---

## 1. ¿Qué problema resuelve TA-GAT?

El objetivo de TA-GAT es **inferir redes de regulación génica (GRN)** a partir de datos de expresión genómica (RNA-Seq o microarray).

Una **red de regulación génica** es un mapa de interacciones entre genes: quién activa a quién, quién inhibe a quién, etc. Conocerla nos permite entender qué cambia molecularmente en una enfermedad (por ejemplo, en un tumor respecto a tejido normal).

El problema es que medir directamente estas interacciones en el laboratorio es carísimo. En cambio, tenemos datos de **expresión génica**: medimos cuánto se "activa" (expresa) cada gen en cada muestra. Si dos genes suben y bajan juntos en muchas muestras, quizá están conectados. Pero la correlación no implica causalidad, y los métodos estadísticos clásicos (correlación de Pearson, WGCNA, ARACNE) tienen limitaciones.

**TA-GAT** propone usar una **Graph Attention Network (GAT)** variacional que:
1. Empieza con una red de co-expresión estadística como estructura inicial.
2. Aprende representaciones latentes de los genes teniendo en cuenta el contexto de sus vecinos en el grafo.
3. Incorpora priors biológicos: qué genes son factores de transcripción (TFs), cuál es su significado estadístico en el análisis diferencial.
4. Reconstruye la red aplicando múltiples objetivos de pérdida: precisión en la reconstrucción de aristas, regularización Bayesiana (KL), topología libre de escala y esparsidad.

La hipótesis es que la GNN puede aprender patrones de conectividad más ricos que los métodos estadísticos clásicos, y producir una red más fiel a la biología real.

---

## 2. Visión general del pipeline

```
Datos RNA-Seq / microarray
        |
        v
[ Análisis diferencial (DESeq2) ]
  -> Identifica genes diferencialmente expresados (DEGs)
        |
        v
[ Construcción del grafo de entrada ]
  -> Correlación de Pearson entre DEGs
  -> Boost de pesos para factores de transcripción
  -> Características de nodo: expresión + grado + centralidad + log2FC + padj
        |
        v
[ Entrenamiento GNN (TA-GAT) ]
  -> Encoder GATv2 variacional (2 capas de atención + 2 cabezas de salida)
  -> Loss: reconstrucción (BCE) + KL + Scale-Free + Sparsity
  -> Early stopping
        |
        v
[ Inferencia de la red ]
  -> Similitud coseno en el espacio latente
  -> Top-K aristas por densidad objetivo (5%)
        |
        v
[ Evaluación contra gold standards ]
  -> BioGRID (interacciones físicas experimentales)
  -> STRING (redes funcionales de proteínas)
  -> GeneMANIA (co-expresión curada)
  -> Métricas: AUC-ROC, AUPRC, Precision@K, F1
        |
        v
[ Análisis biológico ]
  -> Identificación de hubs (genes más conectados)
  -> Red diferencial (qué conexiones cambian entre condiciones)
  -> Análisis funcional: módulos Louvain + GO/KEGG
```

---

## 3. Arquitectura de ficheros

```
TA-GAT/
├── new_implementation/         <- Código fuente principal
│   ├── main.py                 <- Orquestador: ejecuta todo el pipeline
│   ├── config.py               <- Todos los parámetros del sistema
│   ├── interactive_setup.py    <- Configuración interactiva del dataset
│   ├── data_loader.py          <- Descarga de datos, DESeq2, construcción del grafo
│   ├── model_gnn.py            <- Arquitectura de la red neuronal TA-GAT
│   ├── metrics.py              <- Métricas de evaluación y visualizaciones
│   ├── biological_relevance.py <- Hubs, red diferencial
│   └── functional_analysis.py  <- Módulos Louvain y enriquecimiento GO/KEGG
│
├── data/                       <- Bases de datos descargadas automáticamente
│   ├── TF_names_v_1.01.txt     <- Lista de factores de transcripción humanos (~1600)
│   ├── BIOGRID-ORGANISM-Homo_sapiens-*.txt <- Gold standard BioGRID
│   ├── 9606.protein.links.v12.0.txt.gz     <- Gold standard STRING
│   ├── 9606.protein.info.v12.0.txt.gz      <- Mapeo proteínas STRING -> gen
│   ├── genemania_identifier_mappings.txt   <- Mapeo IDs GeneMANIA
│   ├── Co-expression.*.txt                 <- Redes co-expresión GeneMANIA
│   └── Conjuntos a ejecutar/              <- Datasets locales pre-procesados
│       └── <nombre_dataset>/
│           ├── counts_control_CLEAN.csv
│           └── counts_primary_CLEAN.csv
│
├── requirements.txt
└── submit_tagat.sh             <- Script de envío al HPC (Hércules)
```

---

## 4. PASO 0 — Configuración inicial

### `config.py` — Los parámetros que controlan todo

Este archivo centraliza **todos los hiperparámetros y rutas**. La clase `Config` (instanciada como `CFG`) es el panel de control del algoritmo.

**Parámetros de datos:**

| Parámetro | Valor por defecto | Qué hace |
|---|---|---|
| `GEO_ID` | None | Accesión del dataset en GEO (p.ej. `GSE11121`) |
| `GEO_METADATA_KEY` | None | Columna de metadatos para separar grupos |
| `GEO_GROUPS` | {} | Qué valores de metadata corresponden a cada grupo |
| `GEO_CONTROL_GROUP` | None | Grupo de referencia para DESeq2 |
| `LOCAL_MODE` | False | Si es True, carga CSVs locales en vez de GEO |
| `LOCAL_DATASET_PATH` | None | Ruta al directorio con los CSVs locales |
| `MIN_TOTAL_COUNTS_PER_GENE` | 10 | Filtra genes con muy pocas lecturas |
| `ALPHA` | 0.05 | Umbral de p-value ajustado para DEGs |
| `LOG2FC_THRESH` | 0.58 | Cambio mínimo en log2 fold-change para ser DEG |

**Parámetros de la GNN:**

| Parámetro | Valor por defecto | Qué hace |
|---|---|---|
| `GAT_HIDDEN_CHANNELS` | 128 | Dimensión de las capas ocultas de atención |
| `GAT_OUT_CHANNELS` | 32 | Dimensión del espacio latente (embeddings) |
| `GAT_HEADS` | 4 | Número de cabezas de atención paralelas |
| `GAT_DROPOUT` | 0.2 | Dropout en los coeficientes de atención |
| `EPOCHS` | 500 | Máximo de épocas de entrenamiento |
| `LR` | 0.001 | Tasa de aprendizaje (Adam) |
| `PATIENCE` | 50 | Épocas sin mejora antes de early stopping |
| `MIN_DELTA` | 1e-4 | Mejora mínima para resetear la paciencia |
| `N_SEEDS` | [42, 123, 456] | Semillas para el promedio multi-semilla |
| `FAST_MODE` | False | Si True, usa solo 1 semilla (~20 min) |

**Parámetros de la función de pérdida:**

| Parámetro | Valor | Qué hace |
|---|---|---|
| `LAMBDA_RECON` | 5.0 | Peso de la pérdida de reconstrucción (BCE) |
| `LAMBDA_KL` | 0.01 | Peso de la regularización KL (prior gaussiano) |
| `LAMBDA_SCALE_FREE` | 2.0 | Peso de la pérdida topológica libre de escala |
| `LAMBDA_SPARSITY` | 0.3 | Peso de la pérdida de esparsidad |

**Parámetros de grafos y baselines:**

| Parámetro | Valor | Qué hace |
|---|---|---|
| `TF_BOOST_FACTOR` | 1.5 | Amplificación de pesos de aristas con TFs |
| `INPUT_GRAPH_PERCENTILE` | 75 | Percentil de Pearson para construir el grafo de entrada GNN |
| `CORR_THRESHOLD_PERCENTILE` | 95 | Percentil para binarizar la baseline WGCNA |
| `WGCNA_POWER_BETA` | 4 | Exponente de soft-thresholding WGCNA |
| `INFERENCE_TARGET_DENSITY` | 0.05 | Densidad objetivo de la red final (5%) |
| `DYNAMIC_NEG_SAMPLING` | True | Re-muestrear negativos cada época |

---

### `interactive_setup.py` — El asistente de configuración

Esta es la **primera cosa que ejecuta el programa** cuando se lanza `python main.py`. Presenta un menú interactivo para configurar el dataset.

**`run_interactive_setup()`** — La función principal:
- Si la variable de entorno `AUTOMATED=1` está activa (modo automático/CI), configura directamente el dataset GSE11121.
- Si `LOCAL_DATASET=<nombre>` está definido, usa ese dataset local automáticamente.
- En modo interactivo, pregunta al usuario:
  1. ¿Usar CSV local o descargar de GEO?
  2. Si GEO: el ID del dataset (GSE XXXXX)
  3. Qué columna de metadatos usar para separar grupos
  4. Qué valores corresponden al grupo Normal y al grupo Tumor

**`run_local_setup()`** — Para datos pre-procesados:
- Escanea la carpeta `data/Conjuntos a ejecutar/`
- Busca subcarpetas que tengan `counts_control_CLEAN.csv` y `counts_primary_CLEAN.csv`
- El usuario elige cuál ejecutar
- Configura `CFG.LOCAL_MODE=True` y las rutas correspondientes

**`parse_geo_metadata(gse)`** — Para datos GEO:
- Analiza los metadatos de cada muestra (GSM) del dataset
- Extrae pares clave-valor de `characteristics_ch1` (ej: `tissue: tumor`)
- Devuelve un diccionario con todos los valores disponibles para que el usuario elija cómo dividir los grupos

---

## 5. PASO 1 — Carga y preprocesamiento de datos

### `data_loader.py` — Carga de datos de expresión

#### Ruta A: Datos de GEO — `load_geo_and_run_deseq2()`

1. **Descarga el dataset de GEO** usando la librería `GEOparse` con el ID configurado.
2. **Extrae la matriz de expresión**: intenta leer las tablas de cada muestra individual (GSM). Si están vacías, busca un fichero suplementario (matrix CSV/TSV comprimido).
3. **Traduce IDs Ensembl a símbolos de genes**: si los genes están en formato `ENSG000...`, consulta la API de MyGene.info en lotes de 1000 para convertirlos a nombres legibles (BRCA1, TP53, etc.).
4. **Mapea sondas de microarray a genes**: si hay una plataforma GPL, usa su tabla para convertir IDs de sonda a símbolos de genes. Si hay varios probes por gen, se promedian.
5. **Ejecuta DESeq2** (análisis estadístico diferencial):
   - Filtra genes con suma de lecturas < 10 en todas las muestras
   - Si detecta valores negativos (expresión en log), los revierte a cuentas crudas (2^x)
   - Redondea las cuentas a enteros (requisito de DESeq2)
   - Contrasta cada grupo caso vs control (Tumor vs Normal)
   - Extrae los **DEGs**: genes con `padj <= 0.05` y `|log2FC| >= 0.58`
   - Si hay más de 1500 DEGs, se quedan los 1500 más significativos (para controlar la memoria)

#### Ruta B: Datos locales — `load_local_counts_and_run_deseq2()`

Hace exactamente lo mismo que la ruta A pero partiendo de los ficheros CSV locales. Une ambas matrices (control + tumor), construye los metadatos de muestra y ejecuta el mismo DESeq2.

**Ambas devuelven:**
- `sig_degs`: DataFrame con los DEGs y sus estadísticos (log2FC, padj, etc.)
- `expr_groups_dict`: diccionario `{nombre_grupo -> matriz_expresión}` donde la matriz tiene shape `(num_DEGs x num_muestras_del_grupo)`

---

#### Carga de gold standards

**`download_and_parse_biogrid()`**:
- Descarga el zip de BioGRID (~40 MB) si no está en local
- Extrae el fichero tab3 específico de Homo sapiens
- Filtra solo interacciones donde ambos organismos son humanos (TaxID 9606)
- Elimina bucles propios (gen que interactúa consigo mismo)
- Convierte a mayúsculas y ordena los pares alfabéticamente para evitar duplicados
- Devuelve un conjunto de tuplas `{(gen1, gen2), ...}` con todas las interacciones verificadas experimentalmente

**`download_and_parse_string()`**:
- Descarga la base de datos STRING de interacciones proteína-proteína (~80 MB)
- Descarga el fichero de información de proteínas para traducir IDs ENSP a símbolos de genes
- Filtra solo interacciones con score >= 400 (umbral configurable, escala 0-1000)
- Mismo proceso de limpieza que BioGRID

**`download_and_parse_genemania_coexp()`**:
- Descarga el fichero de mapeo de IDs de GeneMANIA
- Descarga 3 redes de co-expresión curadas (Wang-Maris, Mallon-McKay, Roth-Zlotnik)
- Mapea IDs propietarios de GeneMANIA a símbolos de genes HUGO

**`create_ground_truth_adj(genes, gs_edges)`**:
- Toma la lista de DEGs y el conjunto de aristas del gold standard
- Crea una **matriz de adyacencia binaria simétrica** de tamaño (N_DEGs x N_DEGs)
- También extrae el conjunto de **"genes evaluables"**: la intersección entre los DEGs y los genes que aparecen en el gold standard. Esta intersección es crucial para la evaluación justa: no tiene sentido penalizar al modelo por no predecir interacciones entre genes que el gold standard ni siquiera conoce.

**`load_tf_list(path)`**:
- Lee el fichero `TF_names_v_1.01.txt` con ~1600 nombres de factores de transcripción humanos conocidos
- Devuelve un conjunto para búsquedas rápidas

---

## 6. PASO 2 — Construcción del grafo de entrada para la GNN

### `build_correlation_baseline(expr, tf_set, degs_df)` — en `data_loader.py`

Esta función hace dos cosas a la vez: construye los **baselines estadísticos** de comparación y prepara el **grafo de entrada** para la GNN.

#### 1. Matrices de correlación

- **Pearson**: `corr_p = |corrcoef(expr.values)|`. Mide correlación lineal. Valor absoluto para capturar tanto co-expresión positiva como negativa.
- **Spearman**: Se calculan los rangos de cada gen por muestra y se aplica Pearson sobre los rangos. Más robusta a outliers.

#### 2. WGCNA soft-thresholding (solo para la baseline WGCNA)

Se eleva la correlación al exponente `beta=4`: `corr_p_powered = corr_p ** 4`. Esto amplifica diferencias entre correlaciones altas (0.9^4 ≈ 0.66) y bajas (0.3^4 ≈ 0.008), reduciendo el ruido de fondo.

#### 3. TF Prior Boosting

Si al menos uno de los dos genes en un par es un factor de transcripción conocido, su correlación se multiplica por `TF_BOOST_FACTOR=1.5`. La lógica: los TFs son reguladores master, y sus interacciones tienen más probabilidad de ser reales. Esto inyecta conocimiento biológico sin forzar ninguna arista concreta.

#### 4. Binarización estandarizada de baselines

Todos los baselines (Pearson, Spearman) se binarizan seleccionando las **top-K aristas** donde K corresponde a una densidad del 5% del grafo completo. Esto garantiza que las comparaciones sean justas: todos los modelos tienen el mismo número de aristas.

#### 5. Grafo de entrada para la GNN

El grafo que recibe la GNN se construye desde `|Pearson| boosted` al **percentil 75** (configurable via `INPUT_GRAPH_PERCENTILE=75`).

> **¿Por qué el percentil 75 y no el 95?** Una versión anterior usaba el mismo umbral del 95 que WGCNA, lo que creaba un problema circular: la GNN recibía exactamente las aristas que WGCNA considera válidas, y solo podía aprender a reconstruir WGCNA. Con el percentil 75, la GNN tiene acceso a un grafo más denso, incluyendo aristas que WGCNA suprimiría, dándole la oportunidad de descubrir conexiones genuinas que los métodos estadísticos pierden.

#### 6. Características de nodo (features del GNN)

Cada nodo (gen) tiene un vector de características:

| Feature | Cómo se calcula | Qué representa |
|---|---|---|
| Expresión normalizada | z-score por gen | Perfil de expresión del gen en todas las muestras |
| Grado normalizado | grado en el grafo de entrada / grado máximo | Cuántas conexiones tiene en el grafo inicial |
| Centralidad de vector propio | `scipy.sparse.linalg.eigsh` (ARPACK) | Importancia del gen en la red de co-expresión |
| log2FoldChange (si disponible) | z-score de los valores de DESeq2 | Magnitud del cambio de expresión entre condiciones |
| -log10(padj) (si disponible) | z-score de `-log10(p-value ajustado)` | Significancia estadística del cambio |

Shape final del tensor de features: `(N_genes, num_samples + 2)` sin DESeq2, o `(N_genes, num_samples + 4)` con DESeq2.

#### 7. Objeto PyG Data

Se crea un objeto `Data` de PyTorch Geometric con:
- `x`: la matriz de features de nodos
- `edge_index`: los índices de aristas del grafo de entrada
- `edge_attr`: los pesos de las aristas (correlación de Pearson boosted)

---

## 7. PASO 3 — La Red Neuronal: arquitectura TA-GAT

### `model_gnn.py` — Las tres piezas del modelo

#### Clase `TopologyAwareGATEncoder` — El codificador

Es el corazón del modelo. Toma el grafo con sus características y produce un **espacio latente** para cada gen.

Usa **GATv2Conv** (Graph Attention Network versión 2), que mejora al GAT clásico calculando la atención en función de la combinación de ambos nodos (no solo del receptor), dándole mayor expresividad.

**Arquitectura (4 capas en total):**

```
Input: x (N x in_channels),  edge_index,  edge_attr (pesos)
  |
  v
[GATv2Conv Layer 1]  ->  ReLU
  4 cabezas de atención, cada una con 128 dimensiones
  Salida: N x (128 x 4) = N x 512
  |
  v
[GATv2Conv Layer 2]  ->  ReLU
  Mismo esquema: 4 cabezas x 128 dim
  Salida: N x 512
  (Añadida para ampliar el campo receptivo a 2-hops)
  |
  +---> [GATv2Conv -> mu]
  |       1 cabeza, 32 dimensiones
  |       Salida: N x 32 (media de la distribución latente)
  |
  +---> [GATv2Conv -> log_std]
          1 cabeza, 32 dimensiones
          Salida: N x 32 (log desviación estándar de la distribución latente)
```

**¿Qué son las cabezas de atención?** Cada cabeza aprende un conjunto diferente de pesos. Tener 4 cabezas paralelas permite al modelo aprender 4 tipos diferentes de relaciones entre genes simultáneamente, luego se concatenan. Es análogo al multi-head attention del Transformer.

**¿Qué son mu y log_std?** En vez de producir directamente un embedding para cada gen, el encoder produce los **parámetros de una distribución gaussiana**. Esto convierte al modelo en un Variational Autoencoder (VAE).

**¿Por qué 2-hops?** La capa 1 agrega información de los vecinos directos (1-hop). La capa 2 agrega información de los vecinos de los vecinos (2-hop). Esto permite que cada gen "sepa" algo sobre genes que no están directamente conectados a él pero sí conectados a sus vecinos.

---

#### Clase `TAGAT` — El autoencoder variacional

Envuelve al encoder y añade la lógica variacional.

**`encode(x, edge_index, edge_attr)`**:
- Llama al encoder -> obtiene `(mu, logstd)`
- Aplica el **truco de reparametrización**: `z = mu + epsilon * exp(logstd)` donde `epsilon ~ N(0,1)`
- Durante el entrenamiento: z es aleatorio (muestreado de la gaussiana)
- Durante la inferencia: z = mu (determinístico, sin ruido)

**¿Por qué el truco de reparametrización?** Si z es aleatorio, no puedes calcular gradientes a través de él. Al separarlo como `z = mu + epsilon*std`, el gradiente fluye a través de `mu` y `std`, que son salidas deterministas de la red. `epsilon` es ruido externo que no necesita gradiente.

**`kl_loss()`**:
- Calcula la divergencia KL entre la distribución aprendida `N(mu, sigma)` y una gaussiana estándar `N(0, 1)`
- Fórmula: `KL = -0.5 * mean(sum(1 + 2*logstd - mu^2 - exp(logstd)^2, dim=1))`
- Fuerza al espacio latente a estar "organizado" cerca del origen, evitando colapsos donde todos los genes caen en el mismo punto

---

#### Función `ta_gat_loss()` — La función de pérdida compuesta

**1. Pérdida de reconstrucción (Binary Cross-Entropy)**

```
pos_pred = sigmoid(cosine_similarity(z_i, z_j))  para aristas positivas (existen)
neg_pred = sigmoid(cosine_similarity(z_i, z_j))  para aristas negativas (muestreadas, no existen)
recon_loss = -log(pos_pred) - log(1 - neg_pred)
```

Los embeddings se normalizan L2 antes del producto punto (= similitud coseno). Esto asegura que la métrica de entrenamiento es idéntica a la de inferencia.

**Negative sampling**: Por cada arista real (positiva) en el grafo, se muestrea aleatoriamente una arista falsa (par de nodos no conectados). Esto enseña al modelo a distinguir entre pares conectados y los que no.

**2. Pérdida de topología libre de escala** (`scale_free_loss_v2`)

Las redes biológicas reales siguen una distribución de ley de potencia en los grados: pocos genes con muchas conexiones (hubs) y muchos genes con pocas. Esta pérdida fuerza a que la red predicha tenga esa propiedad.

¿Cómo se calcula?
1. Se calcula el "grado suave" de cada gen: suma de probabilidades de arista predichas (valores continuos, no binarizados).
2. Se ordenan los grados de mayor a menor.
3. Se hace una regresión lineal en espacio log-log: `log(grado) ~ log(rango)`. Si hay ley de potencia, esto debería ser lineal con pendiente negativa.
4. La pérdida penaliza:
   - `1 - R²`: cuando la distribución no es lineal en log-log
   - `max(0, pendiente + 0.3)`: cuando la pendiente es positiva (la distribución crece en vez de decrecer)
   - `1 / (CV + 0.1)`: cuando todos los genes tienen grados similares (poca heterogeneidad)

> **Nota técnica**: Los grados suaves se calculan directamente sobre las aristas del grafo de entrenamiento (O(E)), no sobre la matriz densa NxN completa (O(N^2)). Esto es mucho más eficiente.

**3. Pérdida de esparsidad dirigida** (`targeted_sparsity_loss`)

Penaliza dos cosas:
- Que el gen con **menos conexiones** tenga un grado suave alto
- Que el gen **mediano** tenga un grado suave alto

La idea: permitir que algunos genes sean hubs muy conectados, pero que la mayoría tengan pocas conexiones. Fomenta una estructura tipo escala-libre.

**La pérdida total:**

```
Loss = 5.0 * recon_loss
     + 0.01 * (1/N) * kl_loss
     + 2.0 * sf_loss
     + 0.3 * sparsity_loss
```

---

## 8. PASO 4 — Entrenamiento de la GNN

### `train_ta_gat()` — en `main.py`

**Inicialización del modelo:**
```python
encoder = TopologyAwareGATEncoder(
    in_channels = num_features_por_gen,   # samples + 2 (o +4 con DESeq2)
    hidden_channels = 128,
    out_channels = 32,
    heads = 4,
    dropout = 0.2
)
model = TAGAT(encoder)
optimizer = Adam(lr=0.001)
```

**Bucle de entrenamiento (por época):**
1. Limpia gradientes del paso anterior
2. **Negative sampling dinámico**: se muestrean aristas negativas nuevas cada época. Usar negativos frescos cada vez evita que el modelo memorice un conjunto fijo de negativos.
3. Forward pass: `z = model.encode(...)` -> produce el espacio latente z de shape (N_genes, 32)
4. Calcula las tres pérdidas: `recon_loss, sf_loss, sparsity_loss = ta_gat_loss(...)`
5. Calcula la divergencia KL: `kl_div = (1/N) * model.kl_loss()`
6. Combina pérdidas: `loss = 5*recon + 0.01*kl + 2*sf + 0.3*sparsity`
7. Backpropagation: `loss.backward()` calcula gradientes
8. `optimizer.step()` actualiza los pesos de la red

**Early stopping:**
- Si la pérdida no mejora en más de `MIN_DELTA=1e-4` durante `PATIENCE=50` épocas consecutivas, el entrenamiento para.
- Esto previene el sobreajuste y ahorra tiempo cuando el modelo ya ha convergido.

**Output:** Lista de pérdidas por época + gráfico de la curva de aprendizaje guardado como PNG.

---

## 9. PASO 5 — Inferencia de la red final

### `infer_network()` — en `main.py`

Una vez entrenado el modelo, se infiere la red:

1. El modelo entra en **modo evaluación** (`model.eval()`): sin dropout, z = mu directamente (sin ruido).
2. Se produce el embedding `z` de shape (N_genes, 32).
3. Se normalizan los embeddings L2: `z_norm = z / ||z||`
4. Se calcula la **matriz de similitud coseno**: `S = sigmoid(z_norm @ z_norm.T)` donde `S[i,j]` es la probabilidad predicha de que los genes i y j estén conectados.
5. Se eliminan los bucles propios: `S[i,i] = 0`
6. Se seleccionan las **top-K aristas** de la mitad superior de la matriz, donde `K = N*(N-1)/2 * 0.05` (5% de densidad objetivo).
7. Se crea la **matriz binaria de adyacencia** simétrica final.

> **¿Por qué similitud coseno?** Porque el entrenamiento también usa similitud coseno en la BCE loss. Alinear la métrica de entrenamiento con la de inferencia fue una corrección importante que mejoró la consistencia del pipeline.

---

## 10. PASO 6 — Baselines de comparación

Se calculan cuatro métodos de referencia para comparar con TA-GAT:

### Pearson y Spearman
Ya calculados en `build_correlation_baseline()`. Se binarizan con top-K igual que TA-GAT.

### ARACNE — `build_aracne_baseline()` en `data_loader.py`

ARACNE (Algorithm for the Reconstruction of Accurate Cellular Networks) es un algoritmo clásico de inferencia de redes basado en **información mutua**.

1. **Calcula la MI**: `MI = -0.5 * ln(1 - r^2)`. A mayor correlación, mayor información mutua.
2. **Aplica el Data Processing Inequality (DPI)**: Si el gen A comparte más información con C vía B que directamente, la conexión A-C es probablemente indirecta. ARACNE elimina el enlace más débil en cada triángulo.
   - **Implementación vectorizada (FIX)**: En vez de un triple bucle Python (O(N^3), que tardaba 4-6 horas), se usa broadcasting NumPy sobre N matrices NxN. Reduce el tiempo a ~20 segundos para N=1500.
3. Se binariza con top-K a densidad del 5%.

### WGCNA — `build_wgcna_baseline()` en `data_loader.py`

WGCNA (Weighted Gene Co-expression Network Analysis):
1. Calcula `|Pearson|`
2. Eleva al exponente beta=4: `adj = |r|^4`
3. Binariza con top-K al 5% de densidad

---

## 11. PASO 7 — Evaluación y métricas

### `metrics.py` — Cómo se mide si el modelo funciona

#### Evaluación "Fair" — `evaluate_against_gold_standard()`

El gold standard no cubre todos los genes. Por ejemplo, BioGRID conoce interacciones de solo algunos miles de genes. Si tu red predice interacciones entre genes que BioGRID ni siquiera ha estudiado, eso no debería contarse como error.

La evaluación "fair" resuelve esto:
1. Se calcula la **intersección** entre los DEGs del estudio y los genes en el gold standard ("genes evaluables")
2. Se extrae la **sub-red** de solo esos genes en la predicción y el gold standard
3. Se calculan métricas solo en esa sub-red

**Métricas calculadas:**

| Métrica | Qué mide |
|---|---|
| **Precision** | De todas las aristas predichas, ¿qué fracción es real? TP/(TP+FP) |
| **Recall** | De todas las aristas reales, ¿qué fracción se predijo? TP/(TP+FN) |
| **F1-Score** | Media armónica de Precision y Recall |
| **Accuracy** | Fracción de aristas correctamente clasificadas |

#### AUC-ROC — `compute_auc_against_gs()`

Usa los **scores continuos** (antes de aplicar el umbral). Para cada posible umbral calcula TPR y FPR; el AUC es el área bajo esa curva. Mide la capacidad de rankeado sin depender de un umbral concreto.

> El AUC puede ser optimista en redes muy desbalanceadas (95% de pares no conectados). Por eso también se calcula AUPRC.

#### AUPRC — `compute_auprc_against_gs()`

Area Under the Precision-Recall Curve. Más informativa que AUC cuando el dataset está muy desbalanceado. GENIE3, SCENIC+ y otros benchmarks modernos de GRN usan AUPRC como métrica primaria.

#### Precision@K — `compute_precision_at_k()`

De las top-K predicciones más confiadas, ¿cuántas son reales? Se calcula para K in {50, 100, 200, 500}. Directamente interpretable: un biólogo que va a validar 100 predicciones experimentalmente quiere saber cuántas serán correctas.

#### Métricas de topología — `compute_network_topology_metrics()`

Para cada red (predicha y gold standard):
- **Nodos y aristas**
- **Grado medio**
- **Coeficiente de clustering**: ¿cuánto tienden los vecinos de un nodo a estar también conectados entre sí?
- **Diámetro** del componente conectado más grande: estimado con el heurístico "double-sweep" (2 BFS) en vez del exacto (N BFS), mucho más rápido.
- **Scale-Free R²**: ajuste de la distribución de grados a una ley de potencia en log-log. Cuanto más alto, más parecida a una red biológica real.

#### Test de Wilcoxon — `wilcoxon_pairwise_test()`

Cuando se usan múltiples semillas, se tiene un valor de AUC/AUPRC por semilla para cada modelo. El test de Wilcoxon signed-rank comprueba si TA-GAT tiene valores significativamente mayores que cada baseline (test unilateral). Se necesitan >= 3 semillas; con 5+ se obtiene potencia estadística adecuada para publicaciones.

---

### `run_multi_seed_pipeline()` — El promedio multi-semilla

Para garantizar que los resultados no dependen de una inicialización aleatoria concreta:

1. Para cada semilla: inicializa el modelo de cero, entrena, infiere, evalúa
2. **Promedio de métricas**: media y desviación estándar de AUC, AUPRC, F1, etc. sobre todas las semillas
3. **Red ensemble de consenso**: se promedian las matrices de score de todas las semillas y se binariza el resultado. Produce una red más estable que cualquier ejecución individual.

---

## 12. PASO 8 — Análisis biológico post-inferencia

### `biological_relevance.py`

**`analyze_hubs(adj_dict, genes, tf_set, output_dir)`**:
- Para cada condición (Tumor, Normal), calcula el grado de cada gen en la red final
- Extrae los 20 genes más conectados (hubs)
- Marca cuáles son factores de transcripción conocidos
- Exporta CSV con los resultados
- Los hubs que son TFs son especialmente interesantes: sugieren que el modelo ha identificado reguladores clave del programa transcripcional

**`compute_differential_network(adj_dict, genes, output_dir)`**:
- Compara las redes de dos condiciones (ej. Tumor vs Normal)
- Para cada gen calcula: aristas ganadas en tumor, aristas perdidas, grado diferencial neto
- Los genes con mayor grado diferencial absoluto son los más "rewired" entre condiciones: candidatos a reguladores clave de la transición patológica

### `functional_analysis.py`

**`run_functional_analysis(adj_matrix, genes, output_dir)`**:
1. **Detección de módulos Louvain**: agrupa los genes en comunidades (módulos funcionales). Los genes en el mismo módulo tienden a compartir función biológica.
2. **Visualización**: dibuja la red con los nodos coloreados por módulo usando un layout tipo spring.
3. **Enriquecimiento funcional** (para módulos con >= 15 genes): usa `gseapy` + API de Enrichr para buscar qué procesos biológicos (GO Biological Process 2023) y rutas (KEGG 2021 Human) están sobrerrepresentados en cada módulo.

---

## 13. Parámetros del modelo: tabla de referencia

### Hiperparámetros de la GNN

| Parámetro | Valor | Justificación |
|---|---|---|
| `GAT_HIDDEN_CHANNELS = 128` | Dimensión capas ocultas | Aumentado de 64 para mayor capacidad representacional |
| `GAT_OUT_CHANNELS = 32` | Dimensión espacio latente | Bottleneck para capturar relaciones génicas |
| `GAT_HEADS = 4` | Cabezas de atención | Aumentado de 2 para atención multi-relación más rica |
| `GAT_DROPOUT = 0.2` | Dropout en atención | Regularización para datasets pequeños |
| `EPOCHS = 500` | Épocas máximas | El early stopping detiene antes si converge |
| `LR = 0.001` | Tasa de aprendizaje | Estándar para Adam en problemas de grafos |
| `PATIENCE = 50` | Épocas para early stop | Balance convergencia/tiempo |

### Pesos de la función de pérdida

| Peso | Valor | Por qué ese valor |
|---|---|---|
| `LAMBDA_RECON = 5.0` | El más alto | La reconstrucción de aristas es el objetivo principal |
| `LAMBDA_KL = 0.01` | Pequeño | KL suficiente para regularizar sin aplastar la señal de reconstrucción |
| `LAMBDA_SCALE_FREE = 2.0` | Moderado | Fuerza topología biológica sin dominar sobre la reconstrucción |
| `LAMBDA_SPARSITY = 0.3` | Pequeño | Regularización suave hacia redes esparsas |

### Parámetros del grafo de entrada

| Parámetro | Valor | Qué controla |
|---|---|---|
| `INPUT_GRAPH_PERCENTILE = 75` | Percentil del grafo GNN | Cuántas aristas tiene el grafo de entrada |
| `TF_BOOST_FACTOR = 1.5` | Factor de boost TF | Cuánto se amplifican las conexiones con TFs |
| `INFERENCE_TARGET_DENSITY = 0.05` | Densidad objetivo | 5% de todos los pares posibles |
| `WGCNA_POWER_BETA = 4` | Beta de WGCNA | Exponente de soft-thresholding estándar en la literatura |

---

## 14. Flujo de ejecución completo (resumen)

```
python main.py
    |
    +-- run_interactive_setup()
    |       -> Configura dataset (GEO o local), grupos, directorios de salida
    |
    +-- set_seed(123)  ->  Reproducibilidad
    |
    +-- Detección de GPU/MPS/CPU
    |
    +-- load_tf_list()  ->  Carga ~1600 TFs humanos
    |
    +-- load_geo_and_run_deseq2() [o load_local_counts_and_run_deseq2()]
    |       -> Descarga/carga datos de expresión
    |       -> Ejecuta DESeq2 -> obtiene DEGs (max 1500)
    |       -> Devuelve: sig_degs, expr_groups_dict
    |
    +-- download_and_parse_biogrid()  ->  Gold standard BioGRID
    +-- download_and_parse_string()   ->  Gold standard STRING
    +-- download_and_parse_genemania_coexp()  ->  Gold standard GeneMANIA
    |
    +-- create_ground_truth_adj() x3  ->  Matrices de adyacencia para evaluación
    |
    +-- Para cada condición (Tumor, Normal, ...):
            |
            +-- run_multi_seed_pipeline()
                    |
                    +-- Para cada semilla en N_SEEDS=[42,123,456]:
                            |
                            +-- run_condition_pipeline()
                                    |
                                    +-- build_correlation_baseline()
                                    |       -> Pearson, Spearman, WGCNA-powered
                                    |       -> TF boosting
                                    |       -> Features de nodo (expresión + grado + centralidad + DESeq2)
                                    |       -> PyG Data object (grafo de entrada)
                                    |
                                    +-- TopologyAwareGATEncoder
                                    |       -> Conv1 (1-hop) -> ReLU
                                    |       -> Conv2 (2-hop) -> ReLU
                                    |       -> Conv_mu, Conv_logstd
                                    |
                                    +-- TAGAT (wrapper VAE)
                                    |
                                    +-- train_ta_gat()
                                    |       -> max 500 épocas con early stopping
                                    |       -> Loss: 5*BCE + 0.01*KL + 2*SF + 0.3*Sparsity
                                    |       -> Negative sampling dinámico
                                    |
                                    +-- infer_network()
                                    |       -> Embeddings z = mu (modo eval)
                                    |       -> Similitud coseno -> matriz de scores
                                    |       -> Top-K aristas al 5% densidad
                                    |
                                    +-- build_aracne_baseline()  ->  DPI vectorizado
                                    +-- build_wgcna_baseline()   ->  |r|^4
                                    |
                                    +-- evaluate_against_gold_standard() x 5 modelos x 3 GSs
                                    +-- compute_auc_against_gs() / compute_auprc_against_gs()
                                    +-- compute_precision_at_k()
                                    |
                                    +-- plot_roc_curves() / plot_pr_curves()
                                    +-- export adjacency matrices (.tsv)
                    |
                    +-- average_multi_seed_results()  ->  mean +- std
                    +-- Ensemble consensus network (avg scores -> top-K)
                    +-- wilcoxon_pairwise_test()  ->  significancia estadística
    |
    +-- print_final_summary_tables()  ->  Tablas comparativas
    +-- analyze_hubs()               ->  Top 20 genes hub por condición
    +-- compute_differential_network()  ->  Genes más rewired entre condiciones
```

---

## 15. Historial de cambios relevantes

Este historial documenta los cambios técnicos más importantes que han afectado al comportamiento del algoritmo. Se actualiza con cada commit relevante.

---

### Version 2026-07-17

#### FIX: Campo receptivo ampliado a 2-hops
- **Dónde**: `model_gnn.py`, clase `TopologyAwareGATEncoder`
- **Qué**: Se añadió `conv2` (segunda capa GATv2). Antes el encoder solo tenía 1 capa.
- **Por qué importa**: Con 2 capas, cada gen integra información de sus vecinos y de los vecinos de sus vecinos. Crucial para capturar contexto regulatorio más rico.

#### FIX: Consistencia train/infer con similitud coseno
- **Dónde**: `model_gnn.py` (`ta_gat_loss`) y `main.py` (`infer_network`)
- **Qué**: Ambas funciones normalizan L2 antes del producto escalar (= similitud coseno). Antes, el entrenamiento usaba dot-product sin normalizar y la inferencia usaba coseno.
- **Por qué importa**: Inconsistencia entre la métrica de entrenamiento y la de inferencia hace que el modelo optimice una cosa y se evalúe con otra.

#### FIX: Grafo de entrada GNN desacoplado de WGCNA
- **Dónde**: `data_loader.py`, `build_correlation_baseline()`
- **Qué**: El grafo de entrada para la GNN ahora usa |Pearson| al percentil 75, no el filtrado a 95% de WGCNA.
- **Por qué importa**: Si la GNN recibe exactamente las aristas que WGCNA considera buenas, solo puede aprender a copiar WGCNA. Con un grafo más denso (75%), puede descubrir aristas que WGCNA suprime.

#### FIX: Binarización estandarizada (top-K) para todos los modelos
- **Dónde**: `data_loader.py`, `build_correlation_baseline()`
- **Qué**: Pearson, Spearman, ARACNE, WGCNA y TA-GAT usan el mismo criterio top-K al 5% de densidad.
- **Por qué importa**: Antes algunos modelos podían tener más o menos aristas que otros, haciendo las comparaciones no equitativas.

#### FIX: Inferencia pura GNN (sin boost multiplicativo de WGCNA)
- **Dónde**: `main.py`, `infer_network()`
- **Qué**: Se eliminó el multiplicador `Score = GNN x (1 + 2xWGCNA)` que existía en la versión anterior.
- **Por qué importa**: El multiplicador causaba que el modelo no pudiera recuperar aristas que WGCNA suprimía, poniendo un techo en el Recall igual al de WGCNA.

#### FIX: DPI de ARACNE vectorizado (O(N^3) -> O(N^2))
- **Dónde**: `data_loader.py`, `build_aracne_baseline()`
- **Qué**: Se reemplazó el triple bucle Python por broadcasting NumPy sobre N matrices (NxN).
- **Por qué importa**: Para N=1500, el bucle tardaba 4-6 horas. La versión vectorizada tarda ~20 segundos.

#### FIX: Features de nodo de topología sin sesgo de WGCNA
- **Dónde**: `data_loader.py`, `build_correlation_baseline()`
- **Qué**: El grado y la centralidad de vector propio se calculan desde el grafo de entrada neutral (75% percentil), no desde el grafo WGCNA 95%.
- **Por qué importa**: Antes, los features de topología incorporaban el sesgo de WGCNA, con lo que la GNN partía de información pre-filtrada.

#### IMPROVEMENT: Features DESeq2 en nodos del GNN
- **Dónde**: `data_loader.py`, `build_correlation_baseline()`
- **Qué**: Cuando se dispone de resultados de DESeq2, se añaden `log2FoldChange` y `-log10(padj)` normalizados como features adicionales de cada nodo.
- **Por qué importa**: Inyecta directamente la señal de expresión diferencial en el GNN, ayudando al modelo a distinguir genes fuertemente disregulados del ruido.

#### FIX: KL regularization significativa (LAMBDA_KL: 5e-4 -> 0.01)
- **Dónde**: `config.py`
- **Qué**: Se aumentó el peso de la pérdida KL de 0.0005 a 0.01.
- **Por qué importa**: Con KL tan pequeño, el espacio latente degeneraba. Con 0.01, el prior gaussiano tiene efecto real.

#### FIX: Capacidad del modelo aumentada
- **Dónde**: `config.py`
- **Qué**: `GAT_HIDDEN_CHANNELS` 64 -> 128; `GAT_HEADS` 2 -> 4.
- **Por qué importa**: Mayor capacidad representacional para datasets con hasta 1500 genes.

#### FIX: Diámetro estimado con double-sweep BFS
- **Dónde**: `metrics.py`, función `_approx_diameter()`
- **Qué**: En vez de llamar a `nx.diameter()` (O(N*(N+E))), se usan 2 BFS (O(N+E)).
- **Por qué importa**: Para N=1500 y grafos densos, `nx.diameter()` tardaba minutos por llamada.

#### NEW: AUPRC y Precision@K en tablas de resultados
- **Dónde**: `metrics.py`, `main.py`
- **Qué**: Se añadieron `compute_auprc_against_gs()` y `compute_precision_at_k()`.
- **Por qué importa**: AUPRC es más informativa que AUC para tareas desbalanceadas. P@K es directamente interpretable para validación experimental.

#### NEW: Test de Wilcoxon para comparación multi-semilla
- **Dónde**: `metrics.py`, `main.py`
- **Qué**: Se añadió `wilcoxon_pairwise_test()` y `print_wilcoxon_table()`.
- **Por qué importa**: Permite reportar si la superioridad de TA-GAT sobre los baselines es estadísticamente significativa.

#### NEW: Red ensemble de consenso multi-semilla
- **Dónde**: `main.py`, `run_multi_seed_pipeline()`
- **Qué**: En vez de elegir la mejor semilla, se promedian las matrices de score y se binariza el resultado promedio.
- **Por qué importa**: La red resultante es más estable y menos dependiente de una inicialización concreta.

#### NEW: Soporte para datos locales (LOCAL_MODE)
- **Dónde**: `config.py`, `interactive_setup.py`, `data_loader.py`
- **Qué**: Se añadió la opción de cargar datasets pre-procesados desde CSV locales.
- **Por qué importa**: Permite trabajar con datos de TCGA u otras fuentes pre-procesadas sin necesidad de internet o de GEO.

---

*Documento creado el 2026-07-17. Actualizar con cada commit relevante añadiendo una entrada a la sección "Historial de cambios relevantes".*
