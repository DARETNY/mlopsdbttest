FROM python:3.9-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m mluser
USER mluser
WORKDIR /home/mluser/app

COPY --chown=mluser:mluser pyproject.toml uv.lock ./
RUN uv sync --frozen

# Hem eğitim hem de API kodunu kopyala
COPY --chown=mluser:mluser train.py api.py ./

# İMAJ OLUŞTURULURKEN: Modeli bir kez eğit ve model.pkl'yi üret
RUN uv run python train.py

# Dışarıya açılacak portu belirt
EXPOSE 8000

# KONTEYNER ÇALIŞINCA: FastAPI sunucusunu başlat
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]