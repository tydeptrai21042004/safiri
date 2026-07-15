FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY configs ./configs
COPY data ./data
COPY artifacts ./artifacts
COPY reports ./reports
RUN pip install --no-cache-dir .

ENV SAFIRI_PROJECT_ROOT=/app
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

