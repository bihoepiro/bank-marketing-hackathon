# Guía: cómo desplegar este proyecto en tu propia cuenta de GCP

Esta guía es para cualquiera que clone este repositorio y quiera desplegar su **propia copia** de la API en Google Cloud, con su propia cuenta y su propio proyecto (no se reutiliza nada del proyecto original).

No necesitas tocar el modelo ni el código para esto — el modelo ya viene entrenado y empaquetado (`model/`), así que puedes ir directo a Docker + GCP.

## 0. Qué vas a necesitar

- Una cuenta de Google con **facturación habilitada** (Google da crédito gratis para cuentas nuevas; Cloud Run además tiene una capa gratuita mensual).
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) instalado.
- Este repositorio clonado en tu máquina:
  ```bash
  git clone <URL_DE_ESTE_REPO>
  cd bank-marketing-hackathon
  ```

## 1. Probar que todo funciona localmente (antes de tocar GCP)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Abre `http://localhost:8000/docs`. Si ves la documentación de Swagger con 4 endpoints `POST /predict...`, todo está bien. Detén el servidor (`Ctrl+C`) y sigue con Docker.

```bash
docker build -t bank-marketing-api:local .
docker run -p 8080:8080 bank-marketing-api:local
```

Abre `http://localhost:8080/docs`. Si funciona igual que en local, ya puedes pasar a GCP con confianza.

## 2. Autenticarte en Google Cloud

Instala el CLI si no lo tienes, y corre esto en tu propia terminal (esto abre tu navegador para iniciar sesión — nadie más ve tu contraseña ni tu sesión):

```bash
gcloud auth login
```

## 3. Crear tu propio proyecto de GCP

Elige un **ID único** para tu proyecto (letras minúsculas, números y guiones; debe ser único en todo GCP, no solo en tu cuenta). Por ejemplo:

```bash
gcloud projects create mi-bank-marketing-api --name="Bank Marketing API"
gcloud config set project mi-bank-marketing-api
```

Si el ID ya está en uso, `gcloud` te lo dirá — prueba con otro (agrega tu usuario o un número).

## 4. Vincular facturación

Necesitas una cuenta de facturación activa. Para ver cuáles tienes disponibles:

```bash
gcloud billing accounts list
```

Copia el `ACCOUNT_ID` de una cuenta con `OPEN: True` y vincúlala a tu proyecto:

```bash
gcloud billing projects link mi-bank-marketing-api --billing-account=TU_ACCOUNT_ID
```

Si no tienes ninguna cuenta de facturación, créala desde [console.cloud.google.com/billing](https://console.cloud.google.com/billing).

## 5. Habilitar las APIs necesarias

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project=mi-bank-marketing-api
```

## 6. Crear el repositorio de Artifact Registry

Aquí es donde se va a guardar la imagen Docker. Usamos `us-central1` como región (puedes usar otra si prefieres, solo sé consistente en los pasos siguientes):

```bash
gcloud artifacts repositories create bank-marketing-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Repo Docker para la API de Bank Marketing" \
  --project=mi-bank-marketing-api
```

## 7. Construir y subir la imagen con Cloud Build

Desde la raíz del repo (donde está el `Dockerfile`):

```bash
PROJECT_ID="mi-bank-marketing-api"
REGION="us-central1"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/bank-marketing-repo/bank-marketing-api:v1"

gcloud builds submit --tag "$IMAGE" --project="$PROJECT_ID"
```

Esto puede tardar 1-2 minutos. Al final debe decir `STATUS: SUCCESS`.

## 8. Desplegar en Cloud Run

```bash
gcloud run deploy bank-marketing-api \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60 \
  --project="$PROJECT_ID"
```

Al terminar te va a dar una URL pública como:

```
Service URL: https://bank-marketing-api-xxxxxxxxxxxx.us-central1.run.app
```

## 9. Verificar que de verdad funciona (no confiar solo en "Done")

```bash
URL="https://bank-marketing-api-xxxxxxxxxxxx.us-central1.run.app"   # tu URL real

curl "$URL/"
curl "$URL/health"
curl "$URL/model-info"

curl -X POST "$URL/predict" -H "Content-Type: application/json" -d '{
  "age": 41, "balance": 1200, "day": 15, "campaign": 2, "pdays": -1, "previous": 0,
  "job": "technician", "marital": "married", "education": "secondary",
  "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
  "month": "may", "poutcome": "unknown"
}'
```

O más fácil, corre el cliente automático incluido en el repo, que prueba los 4 modelos con 6 casos cada uno:

```bash
python client/test_api.py --url "$URL"
```

Debería terminar con `TOTAL: 24/24 test cases returned HTTP 200 with a prediction.`

También puedes abrir `$URL/docs` en el navegador y probar manualmente desde ahí (ver README, sección 15).

## 10. Si algo falla

- **"container failed to start and listen on PORT"**: no asumas que es el puerto. Revisa los logs primero:
  ```bash
  gcloud run services logs read bank-marketing-api --region us-central1 --project=mi-bank-marketing-api --limit=50
  ```
  Casi siempre el error real está ahí (un import que falla, un archivo que falta, etc.), no el puerto en sí.
- **Error de facturación al habilitar APIs o crear recursos**: confirma que el paso 4 (vincular facturación) se completó.
- **`gcloud` no se reconoce en PowerShell**: usa la ruta completa al ejecutable, por ejemplo:
  ```powershell
  & "C:\Users\<tu_usuario>\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" --version
  ```

## 11. Limpiar recursos (para no seguir pagando)

Si era solo una prueba y quieres borrar todo:

```bash
gcloud run services delete bank-marketing-api --region us-central1 --project=mi-bank-marketing-api
gcloud artifacts repositories delete bank-marketing-repo --location us-central1 --project=mi-bank-marketing-api
```

Cloud Run no cobra nada si el servicio no recibe tráfico (escala a cero), pero Artifact Registry sí cobra por el almacenamiento de la imagen mientras exista.

---

Con esto tienes tu propia copia de la API corriendo en tu propia cuenta de GCP, totalmente independiente del despliegue original. Si quieres reentrenar los modelos con tus propios ajustes antes de desplegar, revisa el README (secciones 4-7 y 11) para el flujo de entrenamiento.
