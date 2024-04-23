FROM --platform=linux/amd64 python:3.10-slim AS development
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    POETRY_VIRTUALENVS_CREATE=0 \
    VIRTUAL_ENV=/.venv \
    PATH="/.venv/bin:/root/.local/bin/:$PATH"
RUN pip install pipx && \
    pipx install poetry && \
    python -m venv ${VIRTUAL_ENV}
WORKDIR /srv
COPY pyproject.toml poetry.lock ./
RUN --mount=type=secret,id=poetry_username \
    --mount=type=secret,id=poetry_password \
    POETRY_HTTP_BASIC_ARTIFACT_USERNAME=$(cat /run/secrets/poetry_username) \
    POETRY_HTTP_BASIC_ARTIFACT_PASSWORD=$(cat /run/secrets/poetry_password) \
    poetry install --no-root --only=main --no-interaction --no-ansi
COPY service ./service
RUN --mount=type=secret,id=poetry_username \
    --mount=type=secret,id=poetry_password \
    POETRY_HTTP_BASIC_ARTIFACT_USERNAME=$(cat /run/secrets/poetry_username) \
    POETRY_HTTP_BASIC_ARTIFACT_PASSWORD=$(cat /run/secrets/poetry_password) \
    poetry install --only=main --no-interaction --no-ansi
RUN python -c "import compileall; compileall.compile_path(maxlevels=10)"
RUN python -m compileall service


FROM --platform=linux/amd64 python:3.10-slim AS release
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    POETRY_VIRTUALENVS_CREATE=0 \
    VIRTUAL_ENV=/.venv \
    PATH="/.venv/bin:$PATH"
WORKDIR /srv
COPY --from=development /.venv /.venv
COPY --from=development /srv /srv


ENTRYPOINT [ "python3", "-m", "awslambdaric" ]
