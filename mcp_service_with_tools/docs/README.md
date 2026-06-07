# Healthcare MCP Service

An MCP (Model Context Protocol) service mounted inside a FastAPI app that exposes
read-only tools over a healthcare database — doctors, patients, appointments,
medical history, and availability.

## Prerequisites

- Python 3.12+
- A running PostgreSQL instance with the `healthcare_saas_db_local` database
  (or use the bundled Docker Compose setup, which provisions one for you)

## Option 1 — Run with Docker Compose (recommended)

This starts both the database and the service in containers.

```bash
cd mcp_service_with_tools
cp .env.example .env
docker compose up --build
```

The service will be available at `http://localhost:8000`.

## Option 2 — Run locally with an existing PostgreSQL database

1. Install dependencies:

   ```bash
   cd mcp_service_with_tools
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```

2. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `DATABASE_URL` to point at your running database, e.g.:

   ```
   DATABASE_URL=postgresql://healthcare_user:DevPassword@localhost:5432/healthcare_saas_db_local
   API_KEY=your-secret-api-key-here
   ```

3. Start the service:

   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

## Verifying it's running

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok"}
```

## Connecting to the MCP endpoint

The MCP server is mounted at `/mcp` and requires an `X-API-Key` header matching
the `API_KEY` value configured in `.env`:

```
POST http://localhost:8000/mcp
X-API-Key: your-secret-api-key-here
```

Point any MCP-compatible client at this URL to discover and call the available
tools (`get_all_doctors`, `get_all_patients`, `get_patients_by_doctor`,
`get_doctors_by_patient`, `get_medical_history`, `get_doctor_availability`).

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Integration tests (real database, real results)

`tests/test_integration.py` connects to the **real** database (via `DATABASE_URL`)
and prints what each tool actually returns — useful for seeing real data rather
than mocked values. Run them with `-s` so the printed output is visible:

```bash
pytest -m integration -s
```

To run only the fast, mocked unit tests (no live DB required):

```bash
pytest -m "not integration"
```

Plain `pytest` (no `-m` filter) runs both — make sure your database is reachable
first.

Every run also generates a self-contained HTML report at `reports/report.html`
(configured in `pytest.ini`). Open it in a browser to see a pass/fail summary
and expandable details — output, duration, and traceback — for each test:

```bash
start reports/report.html      # Windows
# open reports/report.html     # macOS
# xdg-open reports/report.html # Linux
```
