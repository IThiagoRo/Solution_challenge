#Technical Challenge

Solución al *Technical challenge*: un modelo de Machine Learning que decide, transacción por transacción, si aprobar o rechazar un pago, maximizando la **ganancia económica** del negocio (no solo métricas de clasificación) bajo la regla:

- Transacción legítima aprobada → **+25%** del monto.
- Fraude aprobado → **-100%** del monto.
- Transacción rechazada → **0** (ni gana ni pierde).

## 1. La solución y la experimentación (`notebooks/`)

La experimentación y el desarrollo de la solución están documentados en español en los cuatro notebooks de `notebooks/`, en el orden en que deben leerse/ejecutarse:

| Notebook | Contenido |
|---|---|
| [`0_getting_started.ipynb`](notebooks/0_getting_started.ipynb) | Traduce la regla de negocio en una ecuación de beneficio esperado y deriva el **umbral de decisión teórico óptimo** (`p* = ganancia / (ganancia + pérdida) = 0.20`); define el roadmap del resto del proyecto. |
| [`1_EDA.ipynb`](notebooks/1_EDA.ipynb) | Análisis exploratorio del dataset (150,000 transacciones, 45 días, `fraude` con desbalance ~5%). Hipótesis razonadas sobre el significado de cada columna anonimizada (`a`..`p`), patrones de nulos (`o` es MNAR: su ausencia es en sí misma predictiva), relación de `monto` y `score` con el fraude, y un patrón horario fuerte (madrugada duplica la tasa de fraude). |
| [`2_feature_engineering.ipynb`](notebooks/2_feature_engineering.ipynb) | Split **temporal** train/val/test (70/15/15, por `fecha`, para simular producción y evitar leakage). Define e implementa las transformaciones por columna (imputación con mediana + flags de nulidad, `log1p` en variables sesgadas, one-hot para categóricas de baja cardinalidad, frequency encoding para `j` de alta cardinalidad, descarte de `k` por ser un identificador único, features temporales). Todos los estadísticos se ajustan **solo con train**. |
| [`3_modeling.ipynb`](notebooks/3_modeling.ipynb) | Entrena y compara Regresión Logística, Random Forest, XGBoost y LightGBM (desbalance manejado con pesos de clase, sin resampling). El umbral de decisión de cada modelo se **optimiza en validación** maximizando el beneficio total del portafolio (no accuracy/F1), y se contrasta contra los baselines triviales *aprobar todo* / *rechazar todo*. **XGBoost** resulta el modelo ganador; se evalúa una única vez en test y se guarda como artefacto final. |

Estos notebooks son la base del informe técnico que exige el challenge y explican, con el detalle y la evidencia de cada decisión, tanto el modelo resultante como la lógica productiva descrita en la Sección 2.

Los artefactos entrenados que producen (reutilizados por el resto del proyecto) quedan en `artifacts/`:
- `preprocessing_params.pkl` — estadísticos de preprocesamiento (medianas, top países, frecuencias de `j`) + el esquema de columnas de features.
- `final_model.pkl` — el modelo XGBoost entrenado, su umbral de negocio calibrado y el esquema de features esperado.

## 2. Estructura del proyecto y paso a producción

```
Solution_challenge/
├── api/                        # Servicio FastAPI de inferencia en tiempo real (ver Sección 3)
│   ├── main.py
│   └── schemas.py
├── src/
│   ├── core/                   # Logging y excepción custom compartidos por todo el pipeline
│   ├── pipelines/               # preprocessing, training, evaluation, inference
│   └── main.py                  # CLI: entrena o infiere de punta a punta
├── notebooks/                   # 0..3 — experimentación y reporte técnico (Sección 1)
├── data/                        # Dataset crudo del challenge + data/processed/ (train/val/test)
├── artifacts/                   # Modelo y parámetros de preprocesamiento entrenados
├── docs/                        # Enunciado del challenge
├── .github/workflows/ci-cd.yml  # Lint + build/push de la imagen a ECR
├── Dockerfile
├── requirements.txt             # Entorno completo (notebooks, API, dev)
└── requirements-docker.txt      # Dependencias mínimas del pipeline en el contenedor
```

`src/pipelines/` replica en código de producción exactamente lo desarrollado en los notebooks 2 y 3 (`preprocessing.py`, `training.py`, `evaluation.py`), más `inference.py` para puntuar transacciones nuevas con los artefactos ya entrenados. `src/main.py` orquesta ambos flujos (entrenamiento / inferencia por lote) como CLI.

### CI/CD

El workflow [`ci-cd.yml`](.github/workflows/ci-cd.yml) corre en cada push/PR:
- **Lint**: `isort`, `black` y `flake8` en jobs independientes.
- **Build & push**: solo en push a `main` (o disparo manual), y solo si el lint pasa — construye la imagen Docker y la publica en ECR (tags por commit SHA y `latest`). No corre en PRs de forks, para no exponer las credenciales de AWS.

### Cómo ejecutar el pipeline (entrenamiento / inferencia)

**Opción A — directamente en una máquina, con Python**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Entrenamiento: split temporal, fit de preprocesamiento, entrenamiento del modelo
# y evaluación final en test. Lee data/, escribe en artifacts/.
python src/main.py --training

# Inferencia: puntúa transacciones nuevas y crudas con los artefactos ya entrenados.
python src/main.py --inference \
  --input data/new_transactions_sample.csv \
  --output artifacts/predictions.csv
```

Ambos comandos aceptan `--artifacts-dir` para cambiar dónde se leen/escriben `preprocessing_params.pkl` y `final_model.pkl`.

**Opción B — con el contenedor Docker**

El `Dockerfile` empaqueta `src/` y `data/` (no `api/`) y expone el mismo CLI como entrypoint, así que el contenedor sirve para correr el pipeline de entrenamiento/inferencia por lote, no la API (ver Sección 3):

```bash
docker build -t meli-fraud-detection .

# Entrenamiento (CMD por defecto). Monta artifacts/ y logs/ para persistir
# los resultados fuera del contenedor.
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/logs:/app/logs" \
  meli-fraud-detection --training

# Inferencia
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/logs:/app/logs" \
  meli-fraud-detection --inference \
  --input data/new_transactions_sample.csv \
  --output artifacts/predictions.csv
```

## 3. API en tiempo real (FastAPI)

`api/main.py` expone el modelo entrenado como un servicio HTTP para puntuar transacciones al vuelo, reutilizando el mismo `src/pipelines/inference.py` que usa el CLI por lote (misma lógica de negocio en ambos caminos).

### Cómo levantarla

Requiere que `artifacts/final_model.pkl` y `artifacts/preprocessing_params.pkl` ya existan (generados con `--training`, Sección 2):

```bash
pip install -r requirements.txt   # incluye fastapi y uvicorn
uvicorn api.main:app --reload
```

Por defecto lee los artefactos de `artifacts/`; se puede apuntar a otro directorio con la variable de entorno `ARTIFACTS_DIR`. Con el servidor arriba, la documentación interactiva (OpenAPI/Swagger) queda disponible en `http://localhost:8000/docs`.

### Endpoints

| Método y path | Uso |
|---|---|
| `GET /health` | Verifica que el modelo esté cargado; devuelve `model_name` y el `threshold` de negocio en uso. `503` si el servicio arrancó sin poder cargar los artefactos. |
| `POST /predict` | Puntúa **una** transacción cruda (mismos campos `a`..`p`, `fecha`, `monto`, `score` del dataset original). |
| `POST /predict/batch` | Puntúa una **lista** de transacciones en una sola pasada de preprocesamiento/inferencia. |

Ejemplo de request a `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "a": 4, "b": 0.6717, "c": 2201.4, "d": 1.0, "e": 0.0, "f": 10.0,
    "g": "AR", "h": 0, "j": "cat_a349b79", "k": 0.7290250494115859,
    "l": 1619.0, "m": 0.0, "n": 1, "o": "Y", "p": "N",
    "fecha": "2020-04-16T15:22:25", "monto": 21.33, "score": 9
  }'
```

Respuesta:

```json
{
  "transaction_id": 0.7290250494115859,
  "fraud_probability": 0.0123,
  "is_fraud": false,
  "threshold": 0.62,
  "model_name": "XGBoost"
}
```

`is_fraud` es la decisión de negocio (`P(fraude) >= threshold`), usando el umbral que se calibró sobre validación en `3_modeling.ipynb` / `src/pipelines/training.py` — no un umbral fijo de 0.5.
