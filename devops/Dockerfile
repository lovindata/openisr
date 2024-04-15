FROM python:3.11-buster AS backend_builder
WORKDIR /build/backend
COPY backend/pyproject.toml backend/poetry.lock /build/backend/
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 \
    && /root/.local/bin/poetry config virtualenvs.in-project true \
    && /root/.local/bin/poetry install --without dev --no-root --no-cache --no-interaction

FROM node:20.12-slim AS frontend_builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json /build/frontend/
RUN npm install
COPY frontend /build/frontend
RUN npm run build

FROM python:3.11-slim-buster
WORKDIR /app/backend
COPY --from=backend_builder /build/backend /app/backend
COPY --from=frontend_builder /build/frontend/dist /app/frontend/dist
COPY backend /app/backend
ENV OPENISR_PROD_MODE="True"
EXPOSE 8000
ENTRYPOINT [ "/app/backend/.venv/bin/python3", "main.py" ]