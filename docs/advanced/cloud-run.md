# Deploy ChatSpatial MCP on Google Cloud Run

This guide shows how to run ChatSpatial as a **remote MCP server** on Cloud Run, so you can connect from any MCP-capable client and use any LLM/model backend.

---

## Architecture

- **ChatSpatial** runs in Cloud Run as an MCP server over `streamable-http`.
- Your MCP client connects to the Cloud Run HTTPS URL.
- Your chosen model (Claude, OpenAI, Qwen, DeepSeek, etc.) remains configured in the client.

Cloud Run hosts the tools; your client/model hosts the reasoning.

---

## 1) Build and deploy

From repository root:

```bash
PROJECT_ID="your-gcp-project"
SERVICE_NAME="chatspatial-mcp"
REGION="us-central1"

gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1
```

Notes:
- Use more memory/CPU for larger datasets.
- Concurrency `1` is safer for heavy scientific workloads and memory-bound analyses.

---

## 2) Verify the service

```bash
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')
echo "$SERVICE_URL"
```

The server starts with:

```bash
python -m chatspatial server --cloud-run
```

`--cloud-run` automatically forces:
- `transport=streamable-http`
- `host=0.0.0.0`
- `port=$PORT`

---

## 3) Connect MCP clients

### Clients with native remote MCP (HTTP) support

Use your Cloud Run URL as a remote MCP endpoint in your client configuration.

### Clients with only local stdio MCP support

Use an MCP HTTP↔stdio bridge/proxy (for example `mcp-remote`) and point it to the Cloud Run URL.

---

## 4) Production hardening (recommended)

- Prefer authenticated access (remove `--allow-unauthenticated` and use IAM/service-to-service auth).
- Restrict max instances to control cost.
- Add a mounted volume or object storage workflow for input/output data persistence.
- Configure environment variable `CHATSPATIAL_DATA_DIR` to a writable path.
- Add monitoring/alerting on memory and request latency.

---

## 5) Model-agnostic usage

Because ChatSpatial is exposed as MCP, model choice is independent from deployment:
- Claude / Codex / OpenCode
- OpenAI-compatible models
- Qwen / DeepSeek / Doubao (via compatible MCP clients)

As long as the client supports MCP and can reach the Cloud Run endpoint, the same deployed ChatSpatial service can be reused.
