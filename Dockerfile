# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 : build — compile les dépendances Python (nécessite build-essential)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 : image finale — sans toolchain de compilation, utilisateur non-root
# ---------------------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

# libgomp1 : requis à l'exécution par LightGBM/XGBoost (OpenMP)
# curl     : utilisé par le HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copie du code applicatif (data/, ml/, app/, common/, models/.gitkeep, ...)
COPY . .

# Entraînement du modèle au moment du build (comportement par défaut, inchangé) : l'image est
# ainsi autonome et contient directement models/fraud_model.pkl, prêt à l'emploi. Pour un
# déploiement avec volume de persistance monté au runtime (ex. Render, voir render.yaml),
# passer --build-arg SKIP_TRAIN_AT_BUILD=true : l'entraînement se fera alors au premier accès,
# via le mécanisme de secours déjà présent dans common/model_utils.py::charger_artefacts().
ARG SKIP_TRAIN_AT_BUILD=false
RUN if [ "$SKIP_TRAIN_AT_BUILD" != "true" ]; then python ml/train.py; fi

RUN chown -R appuser:appuser /app
USER appuser

ENV PORT=8501
EXPOSE 8501

# Forme shell (pas exec-array) pour permettre l'expansion de $PORT dans la commande curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

ENTRYPOINT ["sh", "-c", "streamlit run app/main.py --server.port=${PORT} --server.address=0.0.0.0"]
