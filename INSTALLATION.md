# ChatSpatial Installation

## Step 1: Create Virtual Environment

```bash
# Python 3.11+ required
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Or use conda
conda create -n chatspatial python=3.12
conda activate chatspatial
```

## Step 2: Install

**Recommended: Use `uv` for fast, reliable installation**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ChatSpatial
uv pip install chatspatial[full]
```

> **Why uv?** ChatSpatial has complex bioinformatics dependencies (squidpy, scanpy, scvi-tools, spatialdata). Standard `pip` may fail with `resolution-too-deep` errors. `uv` has a more powerful resolver that handles this seamlessly.

<details>
<summary>Alternative: pip (may fail on complex dependency trees)</summary>

```bash
pip install --upgrade pip
pip install chatspatial[full]
```

If you see `resolution-too-deep` error, switch to `uv`.
</details>

| Option | Command | Features |
|--------|---------|----------|
| **Full** | `uv pip install chatspatial[full]` | All 60+ methods |
| Standard | `uv pip install chatspatial` | Core methods |

## Step 3: Configure MCP Client

ChatSpatial works with **any MCP-compatible client** — not limited to Claude/Anthropic.

```bash
# Get your Python path
which python  # e.g., /Users/you/venv/bin/python

# Claude Code
claude mcp add chatspatial /path/to/venv/bin/python -- -m chatspatial server

# OpenCode (supports Qwen, DeepSeek, Doubao, etc.)
# See: https://github.com/opencode-ai/opencode
opencode mcp add

# Codex
codex mcp add chatspatial -- /path/to/venv/bin/python -m chatspatial server
```

**Other clients:** See [Configuration Guide](docs/advanced/configuration.md) for Claude Desktop, OpenCode with custom LLMs, and advanced options.

## Step 4: Verify

```bash
python -c "import chatspatial; print('Ready')"
```

Restart your client, then see [Quick Start](docs/quickstart.md) to begin analyzing.

---

## Requirements

- **Python 3.11+** (3.12 recommended)
- **8GB+ RAM** (16GB+ for large datasets)
- **macOS, Linux, or Windows**

---

## Platform Notes

### Windows

**Not available:** SingleR, PETSc

**Use instead:** Tangram, scANVI, CellAssign for annotation; CellRank works without PETSc.

### Python Version Error

If you see `mcp>=1.17.0 not found`:

```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install chatspatial[full]
```

---

## Optional Dependencies

### R Methods

For RCTD, SPOTlight, CARD, scType, Numbat:

```bash
# Install R 4.4+ from https://cran.r-project.org/
Rscript install_r_dependencies.R
```

### STAGATE

```bash
git clone https://github.com/QIFEIDKN/STAGATE_pyG.git
cd STAGATE_pyG && python setup.py install
```

---


## Docker + Cloudflared (remote MCP endpoint)

Use this when you want local Docker hosting but external apps to connect through a Cloudflare tunnel.

```bash
# Build and run the MCP server container (streamable-http transport, port 8080)
docker compose up --build -d

# Verify MCP endpoint is reachable locally
curl -i -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}'

# Verify the OAuth metadata endpoint (required by OpenAI connector)
curl -i http://localhost:8080/.well-known/oauth-protected-resource

# Expose it publicly via Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8080
```

Once cloudflared starts, it prints a public HTTPS URL like `https://xyz.trycloudflare.com`.
Use `https://xyz.trycloudflare.com/mcp` as the remote MCP endpoint in your client (e.g. OpenAI app).

Notes:
- The container uses **streamable-http transport** (`CHATSPATIAL_TRANSPORT=streamable-http`), bound to `0.0.0.0:8080`. The MCP endpoint is `/mcp` (POST).
- OpenAI's MCP connector performs a `GET /.well-known/oauth-protected-resource` before connecting to discover auth requirements. The server responds with a minimal RFC 9728 metadata indicating no auth is required.
- DNS-rebinding protection is automatically disabled when the server binds to `0.0.0.0`, so the cloudflared `Host` header is accepted without triggering "Invalid Host header" errors.
- `docker-compose.yml` includes `host.docker.internal:host-gateway`, so code in the container can reach services running on your host machine using `http://host.docker.internal:<port>`.
- Keep `cloudflared` running while external MCP clients are connected.

### Loading data files when using Docker + Cloudflared

The MCP server runs inside Docker and cannot access your local filesystem.
There are **three ways** to make your data files available for analysis:

#### Option A – OpenAI file attachment → `fetch_data` (no manual upload needed)

When you attach a file in the OpenAI chat, OpenAI stores it internally and
assigns it a `file_id` (e.g. `file-abc123`). The MCP tool `fetch_data` can
download it directly from OpenAI's API using your API key as Bearer token.

Ask the LLM in the chat:

```
Please fetch my uploaded file using:
fetch_data(
  url="https://api.openai.com/v1/files/file-abc123/content",
  filename="filtered_feature_bc_matrix.h5",
  bearer_token="sk-..."
)
```

The LLM will call `fetch_data`, the server downloads the file from OpenAI,
saves it to `/data/filtered_feature_bc_matrix.h5`, and returns the path.
Then immediately call:

```
load_data(data_path="/data/filtered_feature_bc_matrix.h5", data_type="visium")
```

> **How to find the file_id:** In the OpenAI API, after attaching a file in a
> thread, you can retrieve the file_id from the thread's message or via
> `GET https://api.openai.com/v1/files` with your API key.
> Alternatively, ask the LLM "what is the file_id of the file I just uploaded?"
> — it can read it from the message context.

#### Option B – `curl` upload via HTTP (works with any file, any client)

```bash
curl -X POST https://xyz.trycloudflare.com/upload \
     -F "file=@/path/to/filtered_feature_bc_matrix.h5"
# → { "path": "/data/filtered_feature_bc_matrix.h5", "size_bytes": ..., "sha256": "..." }
```

List files already on the server:
```bash
curl https://xyz.trycloudflare.com/upload
```

Then in the MCP chat: `load_data(data_path="/data/filtered_feature_bc_matrix.h5", data_type="visium")`

#### Option C – Copy directly into `./data/` on the host (no rebuild needed)

```bash
# The ./data directory is bind-mounted into the container at /data
cp /path/to/filtered_feature_bc_matrix.h5 ./data/
# Then: load_data(data_path="/data/filtered_feature_bc_matrix.h5", data_type="visium")
```

---

## Help

- [Configuration Guide](docs/advanced/configuration.md) — MCP client setup
- [Cloud Run Deployment](docs/advanced/cloud-run.md) — Run ChatSpatial as remote MCP over HTTPS
- [Troubleshooting](docs/advanced/troubleshooting.md) — Common issues
- [GitHub Issues](https://github.com/cafferychen777/ChatSpatial/issues)
