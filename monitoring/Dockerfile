FROM node:24-alpine AS web
WORKDIR /build/monitoring/web
COPY monitoring/web/package*.json ./
RUN npm ci
COPY monitoring/web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY knee_workspace/ knee_workspace/
COPY monitoring/__init__.py monitoring/__init__.py
COPY monitoring/kneeops/ monitoring/kneeops/
COPY config/workspace.toml config/workspace.toml
COPY --from=web /build/monitoring/kneeops/static/ monitoring/kneeops/static/
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 kneeops && mkdir /state /downloads && chown kneeops:kneeops /state /downloads
USER kneeops
ENV KNEE_WORKSPACE_ROOT=/app KNEEOPS_STATE_DIR=/state PYTHONUTF8=1
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=2)"
CMD ["kneeops", "serve", "--host", "0.0.0.0"]
