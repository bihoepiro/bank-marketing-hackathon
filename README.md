# Bank Marketing ML API

Hackathon 6 — implementación y despliegue de modelos de Machine Learning usando el **Bank Marketing Dataset (UCI)**.

> ¿Quieres desplegar tu propia copia de esta API en tu propia cuenta de GCP? Ve directo a [GUIA_DESPLIEGUE.md](GUIA_DESPLIEGUE.md).

> **Nota de estandarización:** el profesor pidió usar el dataset de **17 variables** (`bank-full.csv`) como estándar de la clase — ese es el dataset **"standard"**, requerido. Adicionalmente, este proyecto implementa el dataset **extendido de 20 variables** (`bank-additional-full.csv`) como comparación opcional (+4 puntos de participación, según el mismo anuncio). Además, la Hackathon exige entrenar y comparar **al menos 2 modelos** (Parte 1) — en vez de desplegar solo el ganador, **los 2 modelos entrenados (Logistic Regression y Random Forest) se sirven para cada uno de los 2 datasets**: 2×2 = **4 modelos, cargados y funcionando en paralelo dentro de la misma API** (ver sección 8).

## 1. Problema

Predecir si un cliente aceptará contratar un **depósito a plazo** tras una campaña de telemarketing bancario:

```text
y ∈ {yes, no}
```

## 2. Dataset

Se usan dos variantes del **Bank Marketing Dataset (UCI)** (https://archive.ics.uci.edu/ml/datasets/bank+marketing):

| | **standard** (requerido) | **extended** (bonus) |
|---|---|---|
| Archivo | `bank-full.csv` | `bank-additional-full.csv` |
| Filas | 45,211 | 41,188 |
| Variables totales | 17 (16 features + `y`) | 21 (20 features + `y`) |
| Features únicas | `balance`, `day` | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `day_of_week` |
| Centinela de `pdays` | `-1` (nunca contactado) | `999` (nunca contactado) |
| Duplicados exactos | 0 | 12 |

Ambos incluyen variables demográficas del cliente (`age`, `job`, `marital`, `education`...) y del contacto (`contact`, `month`, `campaign`, `pdays`, `previous`, `poutcome`). El dataset extendido agrega, además, indicadores macroeconómicos por trimestre/mes que no existen en el dataset estándar.

## 3. Data leakage: por qué se excluyó `duration`

`duration` es la duración en segundos de la última llamada telefónica con el cliente. Se excluyó de **los 4 modelos finales** porque:

- **Solo se conoce después de que la llamada ya terminó.** En producción, al momento de decidir a quién contactar o de estimar la probabilidad de éxito, esa duración todavía no existe.
- Está fuertemente correlacionada con el resultado (llamadas más largas → mayor probabilidad de "yes"), pero esa correlación es un **efecto posterior a la decisión del cliente**, no una causa disponible de antemano. Usarla constituiría **data leakage**.

Esta exclusión se verifica de forma explícita y programática en tres lugares:

1. `src/preprocessing.py`: `assert "duration" not in STANDARD_FEATURES` y `assert "duration" not in EXTENDED_FEATURES` al importar el módulo.
2. `src/train_common.py` (usado por `src/train.py` y `src/train_extended.py`): se repite la misma verificación antes de entrenar.
3. `api/main.py`: los 2 modelos Pydantic de entrada (uno por dataset, compartido por sus 2 algoritmos) tienen `extra="forbid"`, por lo que cualquier request que incluya `duration` es rechazado con **HTTP 422** antes de llegar a cualquiera de los 4 modelos.

## 4. Modelos

Para **cada dataset** (standard y extended) se entrenaron y compararon los mismos **2 algoritmos** de clasificación, ambos con `class_weight="balanced"` por el desbalance de clases (~88% "no" / ~12% "yes") y una búsqueda pequeña de hiperparámetros (`GridSearchCV`, 3-fold):

| Modelo | Hiperparámetros probados |
|---|---|
| Logistic Regression | `C ∈ {0.1, 1.0, 3.0}` |
| Random Forest | `n_estimators ∈ {200, 300}`, `max_depth ∈ {8, 10, 12}`, `min_samples_leaf=10` (fijo) |

Ambos comparten el mismo preprocesamiento por dataset (`ColumnTransformer`: `StandardScaler` en numéricas + `OneHotEncoder` en categóricas), definido una sola vez en `src/preprocessing.py` (`build_pipeline(model, variant=...)`).

**A diferencia de un flujo típico donde solo el modelo ganador llega a producción, aquí los 2 algoritmos de cada dataset se reentrenan con el 100% de los datos y se despliegan**, cumpliendo literalmente "entrenamiento de al menos 2 modelos" (Parte 1) sin descartar ninguno.

### Calibración del umbral de decisión

Con ~88%/12% de desbalance, el umbral por defecto de `.predict()` (0.5) **subestima sistemáticamente la clase "yes"**. En vez de aceptar eso, se calibra el umbral de decisión por candidato:

1. Se separa, dentro del conjunto de entrenamiento, un split adicional de **validación** (nunca toca el test).
2. Se busca, sobre ese split, el umbral que maximiza F1 para la clase "yes" (`precision_recall_curve`).
3. Las métricas de test que se reportan (sección 5) usan ese umbral calibrado, no 0.5 — y cada uno de los 4 modelos desplegados aplica el suyo en cada predicción (`decision_threshold` en su `model_info.json`, aplicado en `src/predict.py`).

Esto es una técnica estándar para clasificación desbalanceada, no una optimización agresiva de hiperparámetros: mejoró F1 y Balanced Accuracy **simultáneamente** en los 4 modelos (ver comparación con umbral 0.5 en `model/<variant>/comparison.json`, campo `all_candidates`).

## 5. Métricas

Evaluadas sobre un conjunto de test hold-out (20%, estratificado, `random_state=42`), **con el umbral calibrado** de cada modelo:

### Dataset standard (bank-full.csv, 17 vars) — n_test=9,043

| Modelo | Umbral | F1 ("yes") | Balanced Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.663 | 0.4505 | 0.6810 | 0.4837 | 0.4216 | 0.7722 |
| **Random Forest (recomendado)** | **0.590** | **0.4732** | **0.7180** | 0.4273 | 0.5302 | 0.7996 |

### Dataset extended (bank-additional-full.csv, 20 vars, bonus) — n_test=8,236

| Modelo | Umbral | F1 ("yes") | Balanced Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.637 | 0.5031 | 0.7529 | 0.4287 | 0.6088 | 0.8003 |
| **Random Forest (recomendado)** | **0.577** | **0.5196** | **0.7622** | 0.4456 | 0.6228 | 0.8136 |

**Las 4 combinaciones son modelos reales, obtenidos ejecutando `python src/train.py` y `python src/train_extended.py`** (ver `model/standard/comparison.json` y `model/extended/comparison.json` para el detalle completo, incluida la matriz de confusión de cada candidato).

El dataset extendido supera al estándar en ambos algoritmos: los indicadores macroeconómicos (`emp.var.rate`, `euribor3m`, `nr.employed`, etc.) aportan señal real sobre el contexto de cada campaña que el dataset de 17 variables no tiene — es la explicación honesta de la diferencia, no un artefacto de tuning. Dentro de cada dataset, Random Forest supera a Logistic Regression en F1 y Balanced Accuracy por un margen mayor al umbral de 0.01 definido de antemano, por lo que **sería el recomendado si solo se pudiera desplegar uno** — pero como la Hackathon pide comparar (no descartar) modelos, ambos quedan expuestos en la API.

## 6. Modelos finales

Los **4 modelos** (2 datasets × 2 algoritmos) se reentrenaron con el **100% de los datos disponibles** de su dataset y se serializaron, junto con todo el preprocesamiento y su umbral calibrado, en:

```text
model/standard/logistic_regression/model.joblib   (8 KB)   + model_info.json (threshold=0.663)
model/standard/random_forest/model.joblib         (~18 MB) + model_info.json (threshold=0.590)
model/extended/logistic_regression/model.joblib   (8 KB)   + model_info.json (threshold=0.637)
model/extended/random_forest/model.joblib         (~18 MB) + model_info.json (threshold=0.577)
model/<variant>/comparison.json                              -- ambos candidatos + cuál se recomendaría
```

`min_samples_leaf=10` se agregó al Random Forest tras observar que sin ese límite el modelo (con `class_weight="balanced"`) generaba árboles innecesariamente grandes (un primer intento con el dataset extendido serializó un `model.joblib` de 144MB) sin mejora real de F1; con el límite, cada Random Forest final pesa ~18MB, se carga rápido y es mucho más fácil de desplegar.

## 7. Pipeline

Cada `model.joblib` es un `sklearn.pipeline.Pipeline` completo (preprocesador + modelo), por lo que la API **nunca reimplementa** el escalado ni el One-Hot Encoding manualmente.

```text
input_dict
   ↓
predict(input_dict, variant, model_key)   (src/predict.py)
   ↓ valida que estén las features del dataset y que NO llegue "duration"
   ↓ arma un DataFrame de 1 fila con las columnas en el orden esperado
model/<variant>/<model_key>/model.joblib   (Pipeline: preprocessor + modelo)
   ↓ predict_proba() + decision_threshold calibrado de ESE modelo (no el 0.5 por defecto)
{"prediction": "yes"|"no", "probabilities": {"no": p0, "yes": p1}}
```

`predict(input_dict, variant, model_key)` (en `src/predict.py`) es la única función que sabe cómo transformar datos crudos en una predicción para cualquiera de los 4 modelos, y es importada directamente por `api/main.py` — la lógica del modelo no está duplicada en ningún otro lugar.

## 8. API

Implementada con **FastAPI**. **Los 4 modelos se cargan una sola vez al iniciar la aplicación** (`lifespan` de FastAPI), nunca se entrenan en un request. Es **un solo servicio** con un endpoint de predicción por modelo:

| Endpoint | Método | Modelo |
|---|---|---|
| `/` | GET | Información básica del servicio y los 4 modelos disponibles |
| `/health` | GET | Health check (usado por Cloud Run) |
| `/model-info` | GET | Metadatos y métricas reales de **standard + random_forest** |
| `/model-info/standard/logistic_regression` | GET | Metadatos y métricas reales de **standard + logistic_regression** |
| `/model-info/extended` | GET | Metadatos y métricas reales de **extended + random_forest** |
| `/model-info/extended/logistic_regression` | GET | Metadatos y métricas reales de **extended + logistic_regression** |
| `/predict` | POST | **standard + random_forest** — endpoint obligatorio de la Hackathon (dataset requerido, modelo recomendado) |
| `/predict/standard/logistic_regression` | POST | **standard + logistic_regression** |
| `/predict/extended` | POST | **extended + random_forest** (bonus) |
| `/predict/extended/logistic_regression` | POST | **extended + logistic_regression** (bonus) |
| `/docs` | GET | Swagger UI (documentación interactiva, con los 4 endpoints de predicción) |

El payload de entrada depende solo del **dataset** (no del algoritmo): los 2 endpoints `standard` comparten el mismo esquema, y los 2 endpoints `extended` comparten el otro.

### Features requeridas

**Endpoints `standard`** (`/predict` y `/predict/standard/logistic_regression`) — 15 features, sin `duration`: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

**Endpoints `extended`** (`/predict/extended` y `/predict/extended/logistic_regression`) — 19 features, sin `duration`: `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`

Cualquier campo faltante, con valor fuera de las categorías válidas, o con el campo extra `duration`, devuelve **HTTP 422** con el detalle del error de validación (Pydantic, `extra="forbid"`), en los 4 endpoints.

## 9. Ejemplo de request

```json
// POST /predict  o  POST /predict/standard/logistic_regression
{
  "age": 41, "balance": 1200, "day": 15, "campaign": 2, "pdays": -1, "previous": 0,
  "job": "technician", "marital": "married", "education": "secondary",
  "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
  "month": "may", "poutcome": "unknown"
}
```

```json
// POST /predict/extended  o  POST /predict/extended/logistic_regression
{
  "age": 41, "campaign": 2, "pdays": 999, "previous": 0,
  "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
  "euribor3m": 4.857, "nr.employed": 5191.0,
  "job": "technician", "marital": "married", "education": "university.degree",
  "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
  "month": "may", "day_of_week": "mon", "poutcome": "nonexistent"
}
```

## 10. Ejemplo de response

Responses reales obtenidos al ejecutar el request de standard contra los 4 endpoints (local, Docker y Cloud Run — idénticos en los tres):

```json
// POST /predict  (standard + random_forest)
{ "prediction": "no", "probabilities": { "no": 0.5907171151013134, "yes": 0.4092828848986871 } }
```

```json
// POST /predict/standard/logistic_regression
{ "prediction": "no", "probabilities": { "no": 0.598517415162261, "yes": 0.401482584837739 } }
```

```json
// POST /predict/extended  (extended + random_forest)
{ "prediction": "no", "probabilities": { "no": 0.7336964139712184, "yes": 0.26630358602878235 } }
```

```json
// POST /predict/extended/logistic_regression
{ "prediction": "no", "probabilities": { "no": 0.6696643052018969, "yes": 0.33033569479810315 } }
```

## 11. Ejecución local

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Entrenar los 4 modelos (2 algoritmos x dataset -> model/<variant>/<model_key>/)
python src/train.py             # standard: logistic_regression + random_forest (requerido)
python src/train_extended.py    # extended: logistic_regression + random_forest (bonus)

# Levantar la API -- carga los 4 modelos y expone los 4 endpoints de predicción
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Luego abrir `http://localhost:8000/docs` (ahí aparecen los 4 endpoints de predicción).

## 12. Docker

Una sola imagen, un solo proceso, los 4 modelos cargados en memoria.

```bash
docker build -t bank-marketing-api:local .
docker run -p 8080:8080 bank-marketing-api:local
```

Verificado localmente:

```text
GET  http://localhost:8080/                                       -> 200 (lista los 4 modelos)
GET  http://localhost:8080/health                                  -> 200 {"status":"ok"}
GET  http://localhost:8080/model-info                               -> 200 (standard + random_forest)
GET  http://localhost:8080/model-info/standard/logistic_regression  -> 200 (standard + logistic_regression)
GET  http://localhost:8080/model-info/extended                      -> 200 (extended + random_forest)
GET  http://localhost:8080/model-info/extended/logistic_regression  -> 200 (extended + logistic_regression)
GET  http://localhost:8080/docs                                     -> 200 (Swagger UI)
POST http://localhost:8080/predict                                  -> 200 (predicción real)
POST http://localhost:8080/predict/standard/logistic_regression     -> 200 (predicción real)
POST http://localhost:8080/predict/extended                         -> 200 (predicción real)
POST http://localhost:8080/predict/extended/logistic_regression     -> 200 (predicción real)
```

Memoria real observada con los 4 modelos cargados y sirviendo tráfico: **~192MB** (`docker stats`), muy por debajo del límite de 512Mi usado en Cloud Run.

`client/test_api.py` se ejecutó con éxito: **24/24** (6 casos × 4 endpoints) contra el contenedor.

## 13. GCP Deployment

**Proyecto GCP:** `hackathon-6-bank-marketing`
**Región:** `us-central1`
**Artifact Registry:** `bank-marketing-repo`
**Cloud Run:** `bank-marketing-api` — **un solo servicio** que sirve los 4 modelos en paralelo.

Flujo:

```text
Código
  ↓
Docker (Dockerfile, una sola imagen con los 4 modelos)
  ↓
Cloud Build   → gcloud builds submit --tag $IMAGE
  ↓
Artifact Registry  → us-central1-docker.pkg.dev/hackathon-6-bank-marketing/bank-marketing-repo/bank-marketing-api:v5
  ↓
Cloud Run  → gcloud run deploy bank-marketing-api --image $IMAGE --allow-unauthenticated
  ↓
Una API pública con 4 endpoints POST, los 4 modelos activos en paralelo
```

A diferencia del taller de referencia (que descargaba un modelo de Hugging Face en el arranque y dependía de la red), aquí **los 4 modelos viajan empaquetados dentro de la imagen Docker** y se cargan en memoria al iniciar el proceso, por lo que el contenedor no depende de ningún servicio externo para arrancar.

Recursos de Cloud Run: 1 CPU / 512Mi de memoria (suficiente para 4 modelos scikit-learn pequeños servidos con FastAPI — uso real ~192MB), `--allow-unauthenticated` para que la API sea pública.

> Nota de evolución del despliegue: se iteró en 3 diseños hasta llegar a este. Primero, dos servicios Cloud Run separados (uno por dataset, elegido vía variable de entorno). Luego, un servicio único con 2 endpoints (uno por dataset, solo el modelo recomendado). Finalmente, este diseño con **4 endpoints** para no descartar ningún modelo entrenado, cumpliendo el requisito de la Parte 1 ("entrenamiento de al menos 2 modelos") de forma literal en el despliegue, no solo en la comparación offline.

## 14. API pública

```text
https://bank-marketing-api-26693841160.us-central1.run.app
```

Documentación interactiva: https://bank-marketing-api-26693841160.us-central1.run.app/docs

Verificado en vivo (todos los checks del punto 18 del enunciado, ejecutados realmente contra esta URL):

- `GET /` → 200 (lista los 4 modelos)
- `GET /health` → 200 `{"status":"ok"}`
- `GET /model-info`, `/model-info/standard/logistic_regression`, `/model-info/extended`, `/model-info/extended/logistic_regression` → 200 (métricas reales) en los 4
- `GET /docs` → 200
- Los 4 `POST /predict...` → 200 con predicción real
- Los 4 `POST /predict...` con `duration` → 422 (rechazado correctamente)
- `client/test_api.py --url <esta URL>` → **24/24** casos con HTTP 200 (6 por modelo × 4 modelos)
- Logs de Cloud Run revisados: arranque limpio (los 4 modelos cargados), sin errores, todas las requests responden 200/422 según corresponda.

## 15. Pruebas

### Automáticas

```bash
python client/test_api.py --url http://127.0.0.1:8000                                              # los 4 modelos (24 casos)
python client/test_api.py --url https://bank-marketing-api-26693841160.us-central1.run.app           # los 4 modelos, contra Cloud Run
python client/test_api.py --url <url> --model standard-rf   # solo /predict (6 casos)
python client/test_api.py --url <url> --model standard-lr   # solo /predict/standard/logistic_regression (6 casos)
python client/test_api.py --url <url> --model extended-rf   # solo /predict/extended (6 casos)
python client/test_api.py --url <url> --model extended-lr   # solo /predict/extended/logistic_regression (6 casos)
```

Por defecto el script prueba **los 4 endpoints** (24 casos en total: 6 perfiles de cliente distintos por dataset, reutilizados en sus 2 algoritmos — retirado con campaña previa exitosa, estudiante joven nunca contactado, obrero con muchos contactos, gerente con campaña previa fallida, cliente con datos "unknown"/en default, empresario sin contacto previo) e imprime para cada uno: caso, status HTTP, `prediction` y `probabilities`.

### Manuales (para el docente, vía Swagger)

1. Abrir `<URL_PUBLICA>/docs`.
2. Expandir `POST /predict` → **Try it out** → pegar el primer JSON de la sección 9 → Ejecutar.
3. Repetir con `POST /predict/standard/logistic_regression` (mismo JSON).
4. Expandir `POST /predict/extended` → **Try it out** → pegar el segundo JSON de la sección 9 → Ejecutar.
5. Repetir con `POST /predict/extended/logistic_regression` (mismo JSON).
6. Verificar que las 4 respuestas tengan `prediction` y `probabilities` coherentes (no necesariamente iguales entre sí — son 4 modelos distintos).
7. Probar también `GET /`, `GET /health` y los 4 `GET /model-info...` directamente desde el navegador o desde Swagger.
8. Probar el rechazo de leakage: agregar `"duration": 300` a cualquiera de los dos JSON → debe devolver **HTTP 422** en los endpoints correspondientes.

## 16. Nivel de implementación

**Nivel elegido: 1 — Predicción.** (aplica igual a los 4 modelos)

- **Qué hace el sistema:** dado un conjunto de atributos del cliente y del contexto de campaña (sin `duration`), estima la probabilidad de que acepte un depósito a plazo y devuelve una clase (`yes`/`no`) junto con las probabilidades.
- **Qué decisión toma:** ninguna decisión de negocio; solo produce un score/predicción que un humano (equipo de marketing/call center) puede usar como insumo.
- **Qué NO hace:** no decide a quién llamar, no prioriza la lista de contactos, no dispara ninguna acción (no envía campañas, no bloquea/aprueba nada, no se integra con un CRM para ejecutar algo automáticamente).
- **Intervención humana:** total. Un analista o el equipo de marketing interpreta la predicción y decide manualmente si contactar o no al cliente, y cómo.
- **Riesgos:** los 4 modelos tienen un F1 moderado (0.45-0.52) sobre la clase minoritaria "yes" — hay falsos negativos y falsos positivos relevantes (ver matrices de confusión en `model/<variant>/comparison.json`). Usar esta predicción para excluir automáticamente clientes de una campaña, sin revisión humana, sería riesgoso y no es lo que este sistema hace.
- **Para pasar al siguiente nivel (Recomendación):** habría que traducir la probabilidad en una recomendación accionable (p. ej. "contactar" / "no contactar" con un umbral de negocio calibrado, o un ranking de prioridad de la cartera de clientes), validado con el equipo de negocio y con métricas de impacto (no solo F1), pero sin ejecutar ninguna acción automáticamente todavía.

## 17. Entorno realista

### Hackathon (lo que se implementó)

- 4 modelos entrenados offline (2 datasets × 2 algoritmos), servidos en paralelo por una única API REST sin estado.
- Cada modelo se versiona como un archivo (`model.joblib`) dentro de la misma imagen Docker; los 4 se cargan en memoria al iniciar el proceso.
- Sin autenticación en la API (`--allow-unauthenticated`), pensado para que el docente pueda probarla directamente.
- Sin monitoreo ni logging estructurado más allá de los logs por defecto de Cloud Run.
- Despliegue manual vía `gcloud builds submit` + `gcloud run deploy` (un solo servicio).

### Producción (qué evolucionaría)

- **Autenticación/autorización:** proteger los endpoints de predicción con API keys, OAuth2 o IAM de Cloud Run, en vez de acceso público.
- **Gestión segura de secretos:** cualquier credencial (DB, APIs externas) iría en **Secret Manager**, nunca en variables de entorno planas ni en el código.
- **CI/CD:** pipeline (Cloud Build triggers / GitHub Actions) que build, testee y despliegue automáticamente en cada push a `main`, con ambientes separados (staging/prod).
- **Model registry:** versionar los 4 modelos en Vertex AI Model Registry o similar, en vez de archivos `.joblib` en el repo.
- **Monitoreo y observabilidad:** métricas de latencia/errores (Cloud Monitoring) por endpoint, tracing, alertas.
- **Logging estructurado:** registrar requests/predicciones (sin PII sensible) para auditoría y debugging.
- **Model drift / data drift:** monitorear si la distribución de las features de entrada o el desempeño de cada modelo se degrada con el tiempo.
- **Reentrenamiento:** pipeline programado (o disparado por drift) que reentrena y valida los modelos antes de promoverlos a producción — nunca reentrenar dentro de la propia API, como se evitó explícitamente en este proyecto.
- **Decidir entre modelos:** en un escenario real no se mantendrían 4 modelos indefinidamente en paralelo; se elegiría uno (Random Forest + dataset extendido, por mejor desempeño) tras validar que el pipeline que provee los indicadores macroeconómicos esté disponible de forma confiable en producción. Los otros 3 quedarían documentados como comparación, no en servicio activo.

## Arquitectura

```text
                              ┌──────────────┐
                              │    Client    │
                              └──────┬───────┘
                                     │
                                     ▼
                          ┌───────────────────────┐
                          │       FastAPI          │
                          │      Cloud Run         │
                          │  (un solo servicio)    │
                          └────────────┬───────────┘
                                       │
            ┌──────────────┬──────────┴──────────┬──────────────┐
            ▼              ▼                     ▼              ▼
   POST /predict   POST /predict/       POST /predict/  POST /predict/
   (standard+RF)   standard/lr          extended (RF)   extended/lr
            │              │                     │              │
            ▼              ▼                     ▼              ▼
  model/standard/  model/standard/      model/extended/ model/extended/
  random_forest/    logistic_regression/ random_forest/  logistic_regression/
  model.joblib       model.joblib        model.joblib     model.joblib
     (todos cargados en memoria al iniciar el proceso)
```

Flujo de deployment (una imagen, un servicio, 4 modelos en paralelo):

```text
Código
  ↓
Docker
  ↓
Cloud Build
  ↓
Artifact Registry
  ↓
Cloud Run (bank-marketing-api, los 4 modelos cargados al iniciar)
  ↓
API pública con 4 endpoints POST /predict...
```

## Estructura del proyecto

```text
bank-marketing-hackathon/
├── data/                              # datasets (excluidos de la imagen Docker)
│   ├── bank-full.csv                   # standard, 17 vars (requerido)
│   └── bank-additional-full.csv        # extended, 20 vars (bonus)
├── notebooks/
│   ├── modeling_standard.ipynb         # EDA + modelado, dataset standard (ejecutado, outputs reales)
│   └── modeling_extended.ipynb         # EDA + modelado, dataset extended (ejecutado, outputs reales)
├── src/
│   ├── preprocessing.py                # features de ambos datasets, exclusión de duration, ColumnTransformer
│   ├── train_common.py                 # lógica de entrenamiento compartida (grid search, threshold tuning, guarda ambos modelos)
│   ├── train.py                        # entrena standard -> model/standard/{logistic_regression,random_forest}/
│   ├── train_extended.py               # entrena extended -> model/extended/{logistic_regression,random_forest}/
│   └── predict.py                      # predict(input_dict, variant, model_key) reutilizado por la API
├── model/
│   ├── standard/
│   │   ├── logistic_regression/        # model.joblib, metrics.json, model_info.json
│   │   ├── random_forest/              # model.joblib, metrics.json, model_info.json
│   │   └── comparison.json             # comparación de ambos candidatos + modelo recomendado
│   └── extended/                       # misma estructura que standard/
├── api/
│   └── main.py                         # FastAPI: carga los 4 modelos; 4 endpoints POST /predict...
├── client/
│   └── test_api.py                     # cliente automático (24 casos: 6 x 4 modelos, --model para filtrar)
├── Dockerfile                          # una imagen, los 4 modelos cargados en el mismo proceso
├── requirements.txt
├── .dockerignore
├── .gitignore
├── README.md
└── GUIA_DESPLIEGUE.md               # cómo desplegar tu propia copia en tu propia cuenta de GCP
```
