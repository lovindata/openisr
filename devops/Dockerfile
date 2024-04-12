FROM node:20.12-slim AS frontend_builder
WORKDIR /build/frontend
COPY frontend /build/frontend
RUN npm install \
    && npm run build

FROM python:3.11-buster AS backend_builder
WORKDIR /build/backend
COPY backend/pyproject.toml backend/poetry.lock /build/backend/
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 \
    && /root/.local/bin/poetry config virtualenvs.in-project true \
    && /root/.local/bin/poetry install --without dev --no-root --no-cache --no-interaction
COPY backend /build/backend
RUN /root/.local/bin/poetry install --without dev --no-cache --no-interaction

FROM python:3.11-slim-buster
WORKDIR /app
COPY --from=frontend_builder /build/frontend/dist /app/frontend/dist
COPY --from=backend_builder /build/backend /app/backend
WORKDIR /app/backend
ENTRYPOINT [ "/app/backend/.venv/bin/python3", "main.py" ]