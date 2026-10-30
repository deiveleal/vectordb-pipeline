# Imagem da API de busca (src/api.py).
#
# Dependencias pelo uv.lock (--frozen): a imagem tem exatamente as versoes
# testadas. Instaladas antes de copiar o codigo, para a camada ser reusada
# quando so o codigo muda.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

# sem root: a API so le de servicos pela rede
RUN useradd --create-home --uid 10002 api
USER api

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
