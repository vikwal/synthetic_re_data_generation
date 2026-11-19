# Synthetic Renewables dashboard. Build context is the project root (the folder that contains fastapi_app/):
#   docker build -f fastapi_app/Dockerfile -t synthetic-dashboard .
# Served under a sub-path behind the FL server's communicator: set SYNTH_RE_BASE_PATH=/synthetic-dashboard.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only torch first (there is no GPU on the server), then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY fastapi_app/requirements.txt ./fastapi_app/requirements.txt
RUN pip install --no-cache-dir -r fastapi_app/requirements.txt

# Code only (see .dockerignore). The weather/master data (streamlit/data) and users' storage come in through volumes.
COPY . .

RUN useradd --uid 49999 --no-create-home app && mkdir -p /storage && chown -R app:app /storage /app
USER app

EXPOSE 8000
# One worker on purpose: jobs and caches live inside the process. The proxy headers carry the real scheme (https) so
# the login cookie is marked secure.
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
