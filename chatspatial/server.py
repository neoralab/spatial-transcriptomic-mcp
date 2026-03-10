"""
Main server implementation for ChatSpatial using the Spatial MCP Adapter.
"""

import asyncio
import base64
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

# Initialize runtime configuration (SSOT - all config in one place)
# This import triggers init_runtime() which configures:
# - Environment variables (TQDM_DISABLE, DASK_*)
# - Warning filters
# - Scanpy settings
from . import config  # noqa: F401
from .models.analysis import AnnotationResult  # noqa: E402
from .models.analysis import CellCommunicationResult  # noqa: E402
from .models.analysis import CNVResult  # noqa: E402
from .models.analysis import ConditionComparisonResult  # noqa: E402
from .models.analysis import DeconvolutionResult  # noqa: E402
from .models.analysis import DifferentialExpressionResult  # noqa: E402
from .models.analysis import EnrichmentResult  # noqa: E402
from .models.analysis import IntegrationResult  # noqa: E402
from .models.analysis import PreprocessingResult  # noqa: E402
from .models.analysis import RNAVelocityResult  # noqa: E402
from .models.analysis import SpatialDomainResult  # noqa: E402
from .models.analysis import SpatialStatisticsResult  # noqa: E402
from .models.analysis import SpatialVariableGenesResult  # noqa: E402
from .models.analysis import TrajectoryResult  # noqa: E402
from .models.data import AnnotationParameters  # noqa: E402
from .models.data import CellCommunicationParameters  # noqa: E402
from .models.data import CNVParameters  # noqa: E402
from .models.data import ColumnInfo  # noqa: E402
from .models.data import ConditionComparisonParameters  # noqa: E402
from .models.data import DeconvolutionParameters  # noqa: E402
from .models.data import DifferentialExpressionParameters  # noqa: E402
from .models.data import EnrichmentParameters  # noqa: E402
from .models.data import IntegrationParameters  # noqa: E402
from .models.data import PreprocessingParameters  # noqa: E402
from .models.data import RegistrationParameters  # noqa: E402
from .models.data import RNAVelocityParameters  # noqa: E402
from .models.data import SpatialDataset  # noqa: E402
from .models.data import SpatialDomainParameters  # noqa: E402
from .models.data import SpatialStatisticsParameters  # noqa: E402
from .models.data import SpatialVariableGenesParameters  # noqa: E402
from .models.data import TrajectoryParameters  # noqa: E402
from .models.data import VisualizationParameters  # noqa: E402
from .tools.embeddings import EmbeddingParameters  # noqa: E402
from .spatial_mcp_adapter import ToolContext  # noqa: E402
from .spatial_mcp_adapter import create_spatial_mcp_server  # noqa: E402
from .utils.exceptions import ParameterError  # noqa: E402
from .utils.mcp_utils import mcp_tool_error_handler  # noqa: E402

# Create MCP server and adapter
mcp, adapter = create_spatial_mcp_server("ChatSpatial")

# Get data manager from adapter
data_manager = adapter.data_manager


# ---------------------------------------------------------------------------
# RFC 9728 – OAuth 2.0 Protected Resource Metadata
# ---------------------------------------------------------------------------
# OpenAI's MCP connector (and other compliant clients) always perform a
# GET /.well-known/oauth-protected-resource before establishing the MCP
# connection to discover auth requirements.  If the endpoint is absent or
# returns an error the client hangs waiting for auth info that never arrives.
# Since ChatSpatial is deployed without auth (open tunnel via cloudflared),
# we return a minimal valid response advertising *no* authorization servers,
# which tells the client that the resource is publicly accessible.
@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
async def oauth_protected_resource_metadata(request: Request) -> JSONResponse:
    """RFC 9728 protected resource metadata – no auth required."""
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "resource": f"{base_url}/mcp",
            "authorization_servers": [],
            "bearer_methods_supported": [],
            "scopes_supported": [],
        }
    )


# ---------------------------------------------------------------------------
# HTTP file upload endpoint – POST /upload (multipart/form-data)
# ---------------------------------------------------------------------------
# Allows uploading files directly via curl or any HTTP client without going
# through the MCP protocol.  Files are saved to CHATSPATIAL_DATA_DIR (/data)
# and can then be referenced in load_data() as /data/<filename>.
#
# Usage:
#   curl -F "file=@filtered_feature_bc_matrix.h5" \
#        https://xyz.trycloudflare.com/upload
#
# Returns JSON: {"path": "/data/filtered_feature_bc_matrix.h5", "size_bytes": ...}

@mcp.custom_route("/upload", methods=["POST"])
async def http_upload(request: Request) -> JSONResponse:
    """Upload a file via multipart/form-data. Field name: 'file'."""
    try:
        form = await request.form()
    except Exception as exc:
        return JSONResponse({"error": f"Failed to parse form data: {exc}"}, status_code=400)

    upload = form.get("file")
    if upload is None:
        return JSONResponse(
            {"error": "Missing 'file' field. Use: curl -F 'file=@yourfile.h5' /upload"},
            status_code=400,
        )

    # UploadFile-like object from Starlette
    safe_name = Path(getattr(upload, "filename", "") or "upload").name
    if not safe_name:
        return JSONResponse({"error": "Cannot determine filename."}, status_code=400)

    try:
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = _UPLOAD_DIR / safe_name
        file_bytes = await upload.read()

        if len(file_bytes) == 0:
            return JSONResponse({"error": "Uploaded file is empty (0 bytes)."}, status_code=400)
        if len(file_bytes) > _MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": f"File too large: {len(file_bytes) / 1024 / 1024:.1f} MB."},
                status_code=413,
            )

        dest.write_bytes(file_bytes)
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        return JSONResponse({
            "path": str(dest),
            "filename": safe_name,
            "size_bytes": len(file_bytes),
            "sha256": sha256,
            "next_step": f"Call load_data(data_path='{dest}', data_type='visium') to load the dataset.",
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/upload", methods=["GET"])
async def http_upload_list(request: Request) -> JSONResponse:
    """List files already present in the /data directory."""
    try:
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        files = [
            {"filename": f.name, "size_bytes": f.stat().st_size}
            for f in sorted(_UPLOAD_DIR.iterdir())
            if f.is_file() and not f.name.startswith(".")
        ]
        return JSONResponse({
            "data_dir": str(_UPLOAD_DIR),
            "files": files,
            "upload_hint": "POST /upload with form field 'file' to add files.",
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)



P = TypeVar("P")


def _resolve_params(params: Optional[P], default_factory: type[P]) -> P:
    """Resolve optional params to a concrete model instance."""
    return params if params is not None else default_factory()


# ---------------------------------------------------------------------------
# File upload tool – bridge for remote MCP clients (e.g. OpenAI connector)
# ---------------------------------------------------------------------------
# Remote MCP clients cannot reference local filesystem paths because the
# server runs inside Docker.  This tool accepts the file content encoded in
# base64, writes it to the shared /data directory that is bind-mounted from
# the host, and returns the in-container path that load_data() can use.

_UPLOAD_DIR = Path(os.environ.get("CHATSPATIAL_DATA_DIR", "/data"))
_MAX_UPLOAD_BYTES = int(os.environ.get("CHATSPATIAL_MAX_UPLOAD_MB", "500")) * 1024 * 1024
_WIDGET_RESOURCE_DOMAIN = "https://persistent.oaistatic.com"


def _widget_html(title: str, heading: str, payload_var: str) -> str:
    """Generate minimal CSP-safe widget HTML for OpenAI Apps embedding."""
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
      body {{ font-family: Inter, system-ui, -apple-system, sans-serif; margin: 0; padding: 12px; }}
      .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }}
      h3 {{ margin: 0 0 8px; font-size: 14px; }}
      pre {{ margin: 0; white-space: pre-wrap; font-size: 12px; line-height: 1.4; }}
      .muted {{ color: #6b7280; font-size: 12px; margin-top: 8px; }}
    </style>
  </head>
  <body>
    <div class=\"card\">
      <h3>{heading}</h3>
      <pre id=\"payload\">Waiting for tool data…</pre>
      <div class=\"muted\">Rendered from MCP widget template.</div>
    </div>
    <script>
      const data = window.openai?.toolOutput?._meta?.chatspatial?.widget?.{payload_var};
      const el = document.getElementById("payload");
      el.textContent = data ? JSON.stringify(data, null, 2) : "No widget payload found.";
    </script>
  </body>
</html>
"""


@mcp.custom_route("/widgets/spatial-pipeline.html", methods=["GET"])
async def widget_spatial_pipeline(request: Request) -> JSONResponse:
    """Serve OpenAI Apps widget template for spatial pipeline summaries."""
    return JSONResponse(
        {
            "uri": "ui://chatspatial/widgets/spatial-pipeline.html",
            "mimeType": "text/html",
            "text": _widget_html(
                title="ChatSpatial Pipeline Widget",
                heading="Spatial Pipeline Result",
                payload_var="pipeline",
            ),
            "_meta": {
                "openai/widgetDescription": (
                    "Interactive summary for the ChatSpatial pipeline run including preview metadata."
                ),
                "openai/widgetPrefersBorder": True,
                "openai/widgetCSP": {
                    "connect_domains": [],
                    "resource_domains": [_WIDGET_RESOURCE_DOMAIN],
                },
            },
        }
    )


@mcp.custom_route("/widgets/pipeline-status.html", methods=["GET"])
async def widget_pipeline_status(request: Request) -> JSONResponse:
    """Serve OpenAI Apps widget template for async pipeline status."""
    return JSONResponse(
        {
            "uri": "ui://chatspatial/widgets/pipeline-status.html",
            "mimeType": "text/html",
            "text": _widget_html(
                title="ChatSpatial Pipeline Status Widget",
                heading="Pipeline Job Status",
                payload_var="pipeline_status",
            ),
            "_meta": {
                "openai/widgetDescription": "Live-like status card for asynchronous pipeline jobs.",
                "openai/widgetPrefersBorder": True,
                "openai/widgetCSP": {
                    "connect_domains": [],
                    "resource_domains": [_WIDGET_RESOURCE_DOMAIN],
                },
            },
        }
    )


@dataclass
class PipelineJob:
    """Track lifecycle of long-running pipeline jobs."""

    job_id: str
    data_id: str
    status: str = "running"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


_PIPELINE_JOBS: dict[str, PipelineJob] = {}


def _extract_visualization_path(visualization_result: str) -> str:
    """Extract visualization file path from visualize_data output."""
    match = re.search(r"Visualization saved:\s*(.+)", visualization_result)
    if not match:
        raise ParameterError(
            "Unable to extract plot path from visualization result. "
            "Expected text containing 'Visualization saved: <path>'."
        )
    return match.group(1).strip()


def _encode_image_to_data_uri(path: str) -> tuple[str, str]:
    """Load image and return (data_uri, mime_type)."""
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise ParameterError(f"Visualization file not found: {path}")

    ext = image_path.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "svg": "image/svg+xml",
        "webp": "image/webp",
    }.get(ext)
    if not mime:
        raise ParameterError(
            f"Unsupported image format '.{ext}'. Use png/jpg/jpeg/svg/webp output format."
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}", mime


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def fetch_data(
    url: str,
    filename: Optional[str] = None,
    bearer_token: Optional[str] = None,
    context: Optional[Context] = None,
) -> dict:
    """Download a data file from a URL and save it to the server for analysis.

    Works with any HTTP/HTTPS URL, including authenticated OpenAI file URLs.

    To use with an OpenAI uploaded file:
      1. The user uploads a file in the chat (e.g. filtered_feature_bc_matrix.h5)
      2. OpenAI makes it available at:
            https://api.openai.com/v1/files/{file_id}/content
      3. Call fetch_data with that URL and bearer_token=<OpenAI API key>
      4. Use the returned 'path' in load_data()

    Args:
        url: HTTP or HTTPS URL of the file to download.
             For OpenAI files: 'https://api.openai.com/v1/files/file-xxx/content'
        filename: Optional filename to save as (basename only).
                  If omitted, derived from the URL path.
        bearer_token: Optional Bearer token for authenticated URLs.
                      For OpenAI: pass your OpenAI API key (sk-...).
                      Sent as 'Authorization: Bearer <token>' header.

    Returns:
        dict with:
          - path: absolute server-side path to pass to load_data()
          - filename: saved filename
          - size_bytes: downloaded file size in bytes
          - sha256: hex digest for integrity verification
    """
    import urllib.request

    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    if not url.startswith(("http://", "https://")):
        raise ParameterError("url must start with http:// or https://")

    # Derive filename from URL if not provided
    if not filename:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "data_file"
        # Handle OpenAI URL pattern: .../files/file-xxx/content → keep original name unknown,
        # use a generic but recognizable name
        if filename == "content":
            filename = "openai_file"

    safe_name = Path(filename).name
    if not safe_name:
        raise ParameterError(f"Cannot derive a valid filename from '{filename}'.")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / safe_name

    await ctx.info(f"Downloading '{url}' → {dest} …")
    try:
        req = urllib.request.Request(url)  # nosec B310 – URL validated above
        if bearer_token:
            req.add_header("Authorization", f"Bearer {bearer_token}")
        with urllib.request.urlopen(req) as response:  # nosec B310
            file_bytes = response.read()
        dest.write_bytes(file_bytes)
    except Exception as exc:
        raise ParameterError(f"Failed to download '{url}': {exc}") from exc

    size = dest.stat().st_size
    if size == 0:
        dest.unlink(missing_ok=True)
        raise ParameterError(
            f"Downloaded file is empty (0 bytes). The URL may have expired or "
            f"require authentication — pass bearer_token if the URL needs auth."
        )
    if size > _MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise ParameterError(
            f"File too large: {size / 1024 / 1024:.1f} MB exceeds the "
            f"{_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
        )

    sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
    await ctx.info(f"Saved '{safe_name}' ({size:,} bytes, sha256={sha256[:12]}…)")

    return {
        "path": str(dest),
        "filename": safe_name,
        "size_bytes": size,
        "sha256": sha256,
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def upload_data(
    filename: str,
    content_base64: str,
    context: Optional[Context] = None,
) -> dict:
    """Upload a data file as base64-encoded content to the server for analysis.

    Use this tool when you have the raw file content available as a base64
    string. For large files or URL-accessible files, prefer fetch_data() instead.

    Note: OpenAI's file attachments are not automatically base64-encoded into
    tool arguments. If you attached a file in the chat, use fetch_data() with
    the file's download URL, or copy the file into the ./data directory on the
    host and reference it as /data/<filename> in load_data() directly.

    Args:
        filename: Target filename (basename only, no path separators).
        content_base64: File bytes encoded as a base64 string.

    Returns:
        dict with path, filename, size_bytes, sha256.
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    safe_name = Path(filename).name
    if not safe_name or safe_name != filename.replace("\\", "/").split("/")[-1]:
        raise ParameterError(
            f"Invalid filename '{filename}'. Provide a plain filename without directory components."
        )

    try:
        file_bytes = base64.b64decode(content_base64, validate=False)
    except Exception as exc:
        raise ParameterError(f"content_base64 is not valid base64: {exc}") from exc

    if len(file_bytes) == 0:
        raise ParameterError(
            "Decoded file is empty (0 bytes). "
            "OpenAI does not pass file attachments as base64 tool arguments — "
            "use fetch_data(url=...) with the file's download URL instead, "
            "or place the file in ./data/ on the host and call "
            "load_data(data_path='/data/<filename>', ...)  directly."
        )

    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        max_mb = _MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ParameterError(
            f"File too large: {len(file_bytes) / 1024 / 1024:.1f} MB exceeds the "
            f"{max_mb} MB limit. Set CHATSPATIAL_MAX_UPLOAD_MB env var to increase it."
        )

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    await ctx.info(f"Uploaded '{safe_name}' → {dest} ({len(file_bytes):,} bytes)")

    return {
        "path": str(dest),
        "filename": safe_name,
        "size_bytes": len(file_bytes),
        "sha256": sha256,
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def load_data(
    data_path: str,
    data_type: str,
    name: Optional[str] = None,
    context: Optional[Context] = None,
) -> SpatialDataset:
    """Load spatial transcriptomics data with comprehensive metadata profile.

    Args:
        data_path: Path to data file or directory
        data_type: 'visium', 'xenium', 'slide_seq', 'merfish', 'seqfish', or 'generic'
        name: Optional dataset name

    Returns:
        SpatialDataset with cell/gene counts and metadata profiles
    """
    # Create ToolContext for consistent logging
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    await ctx.info(f"Loading data from {data_path} (type: {data_type})")

    # Load data using data manager
    data_id = await data_manager.load_dataset(data_path, data_type, name)
    dataset_info = await data_manager.get_dataset(data_id)

    await ctx.info(
        f"Successfully loaded {dataset_info['type']} data with "
        f"{dataset_info['n_cells']} cells and {dataset_info['n_genes']} genes"
    )

    # Convert column info from dict to ColumnInfo objects
    obs_columns = (
        [ColumnInfo(**col) for col in dataset_info.get("obs_columns", [])]
        if dataset_info.get("obs_columns")
        else None
    )
    var_columns = (
        [ColumnInfo(**col) for col in dataset_info.get("var_columns", [])]
        if dataset_info.get("var_columns")
        else None
    )

    # Return comprehensive dataset information
    return SpatialDataset(
        id=data_id,
        name=dataset_info["name"],
        data_type=dataset_info["type"],  # Use normalized type from dataset_info
        description=f"Spatial data: {dataset_info['n_cells']} cells × {dataset_info['n_genes']} genes",
        n_cells=dataset_info["n_cells"],
        n_genes=dataset_info["n_genes"],
        spatial_coordinates_available=dataset_info["spatial_coordinates_available"],
        tissue_image_available=dataset_info["tissue_image_available"],
        obs_columns=obs_columns,
        var_columns=var_columns,
        obsm_keys=dataset_info.get("obsm_keys"),
        uns_keys=dataset_info.get("uns_keys"),
        top_highly_variable_genes=dataset_info.get("top_highly_variable_genes"),
        top_expressed_genes=dataset_info.get("top_expressed_genes"),
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def preprocess_data(
    data_id: str,
    params: Optional[PreprocessingParameters] = None,
    context: Optional[Context] = None,
) -> PreprocessingResult:
    """Preprocess spatial transcriptomics data (QC, normalization, HVGs, PCA, clustering, spatial neighbors).

    Args:
        data_id: Dataset ID
        params: Preprocessing parameters (all have sensible defaults)
    """
    # Create ToolContext
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import (avoid name conflict with MCP tool)
    from .tools.preprocessing import preprocess_data as preprocess_func

    # Resolve optional wrapper input to concrete params for tool-level contract
    resolved_params = _resolve_params(params, PreprocessingParameters)

    # Call preprocessing function
    result = await preprocess_func(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save preprocessing result
    await data_manager.save_result(data_id, "preprocessing", result)

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def compute_embeddings(
    data_id: str,
    params: Optional[EmbeddingParameters] = None,
    context: Optional[Context] = None,
) -> dict[str, Any]:
    """Compute dimensionality reduction (PCA, UMAP), clustering, and neighbor graphs.

    Args:
        data_id: Dataset ID
        params: Embedding parameters (PCA, UMAP, clustering, etc.)
    """
    from .tools.embeddings import compute_embeddings as compute_embeddings_func

    resolved = _resolve_params(params, EmbeddingParameters)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)
    result = await compute_embeddings_func(data_id, ctx, resolved)
    dumped = result.model_dump()
    await data_manager.save_result(data_id, "embeddings", dumped)
    return dumped


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def run_spatial_pipeline(
    data_id: str,
    cluster_key: str = "leiden",
    n_top_genes: int = 30,
    async_mode: bool = True,
    fast_mode: bool = True,
    context: Optional[Context] = None,
) -> dict[str, Any]:
    """Run PCA → neighbors → UMAP → clustering → marker genes as a pipeline.

    This tool exists for clients that need a single chained call with optional
    asynchronous execution semantics to survive long-running jobs.
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    async def _execute_pipeline() -> dict[str, Any]:
        await ctx.info("Starting pipeline: embeddings + marker genes")

        embedding_params = EmbeddingParameters(
            compute_pca=True,
            compute_neighbors=True,
            compute_umap=True,
            compute_clustering=True,
            clustering_method="leiden",
            clustering_key=cluster_key,
            n_neighbors=10 if fast_mode else 15,
            n_pcs=20 if fast_mode else 30,
            clustering_resolution=0.8 if fast_mode else 1.0,
            umap_min_dist=0.5,
            compute_spatial_neighbors=True,
        )
        embedding_result = await compute_embeddings(
            data_id=data_id,
            params=embedding_params,
            context=context,
        )

        de_params = DifferentialExpressionParameters(
            group_key=cluster_key,
            method="wilcoxon",
            n_top_genes=n_top_genes,
        )
        marker_result = await find_markers(data_id=data_id, params=de_params, context=context)

        viz_result = await visualize_data(
            data_id=data_id,
            params=VisualizationParameters(
                plot_type="feature",
                basis="umap",
                feature=cluster_key,
                output_format="png",
            ),
            context=context,
        )

        plot_path = _extract_visualization_path(viz_result)
        image_uri, mime_type = _encode_image_to_data_uri(plot_path)

        return {
            "data_id": data_id,
            "cluster_key": cluster_key,
            "embedding_result": embedding_result,
            "markers": marker_result.model_dump(),
            "umap_plot_path": plot_path,
            "structuredContent": {
                "status": "completed",
                "data_id": data_id,
                "cluster_key": cluster_key,
                "umap_plot_path": plot_path,
            },
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Pipeline completata: PCA → UMAP → clustering → marker genes. "
                        f"Cluster key: {cluster_key}."
                    ),
                }
            ],
            "_meta": {
                "openai/outputTemplate": "ui://chatspatial/widgets/spatial-pipeline.html",
                "openai/widgetAccessible": True,
                "openai/widgetDescription": "Summary and UMAP preview for completed spatial pipeline.",
                "openai/widgetPrefersBorder": True,
                "openai/widgetCSP": {
                    "connect_domains": [],
                    "resource_domains": [_WIDGET_RESOURCE_DOMAIN],
                },
                "chatspatial/widget": {
                    "pipeline": {
                        "type": "spatial_pipeline",
                        "data_id": data_id,
                        "cluster_key": cluster_key,
                        "umap_plot_path": plot_path,
                        "image": {
                            "mime_type": mime_type,
                            "data_uri": image_uri,
                        },
                    }
                },
            },
        }

    if not async_mode:
        return await _execute_pipeline()

    job_id = f"pipeline_{len(_PIPELINE_JOBS) + 1}"
    _PIPELINE_JOBS[job_id] = PipelineJob(job_id=job_id, data_id=data_id)

    async def _runner() -> None:
        try:
            result = await _execute_pipeline()
            _PIPELINE_JOBS[job_id].status = "completed"
            _PIPELINE_JOBS[job_id].result = result
        except Exception as exc:
            _PIPELINE_JOBS[job_id].status = "failed"
            _PIPELINE_JOBS[job_id].error = str(exc)

    asyncio.create_task(_runner())
    return {
        "job_id": job_id,
        "status": "running",
        "message": "Pipeline started. Poll with get_pipeline_status(job_id).",
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def get_pipeline_status(job_id: str) -> dict[str, Any]:
    """Get status and result of an asynchronous run_spatial_pipeline job."""
    job = _PIPELINE_JOBS.get(job_id)
    if not job:
        raise ParameterError(f"Unknown job_id: {job_id}")

    response: dict[str, Any] = {
        "job_id": job.job_id,
        "data_id": job.data_id,
        "status": job.status,
    }
    if job.status == "completed" and job.result is not None:
        response["result"] = job.result
    if job.status == "failed" and job.error:
        response["error"] = job.error

    response["_meta"] = {
        "openai/outputTemplate": "ui://chatspatial/widgets/pipeline-status.html",
        "openai/widgetAccessible": True,
        "openai/widgetDescription": "Status widget for asynchronous pipeline execution.",
        "openai/widgetPrefersBorder": True,
        "openai/widgetCSP": {
            "connect_domains": [],
            "resource_domains": [_WIDGET_RESOURCE_DOMAIN],
        },
        "chatspatial/widget": {
            "pipeline_status": {
                "job_id": job.job_id,
                "status": job.status,
                "data_id": job.data_id,
                "error": job.error,
            }
        },
    }
    return response


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def visualize_data(
    data_id: str,
    params: Optional[VisualizationParameters] = None,
    context: Optional[Context] = None,
) -> str:
    """Visualize spatial transcriptomics data. Set plot_type and subtype in params; see VisualizationParameters schema for all options.

    Args:
        data_id: Dataset ID
        params: Visualization parameters (plot_type, subtype, genes, output_format, dpi, etc.)
    """
    from .tools.visualization import visualize_data as visualize_func

    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    resolved_params = _resolve_params(params, VisualizationParameters)
    result = await visualize_func(data_id, ctx, resolved_params)

    if result:
        return result
    else:
        return "Visualization generation failed, please check the data and parameter settings."


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def annotate_cell_types(
    data_id: str,
    params: Optional[AnnotationParameters] = None,
    context: Optional[Context] = None,
) -> AnnotationResult:
    """Annotate cell types in spatial transcriptomics data.

    Args:
        data_id: Dataset ID
        params: Annotation parameters (method, reference_data_id, cell_type_key, etc.)

    Note: Reference methods (tangram, scanvi) require reference_data_id to be preprocessed first.
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import annotation tool (avoids slow startup)
    from .tools.annotation import annotate_cell_types

    resolved_params = _resolve_params(params, AnnotationParameters)

    # Call annotation function with ToolContext
    result = await annotate_cell_types(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save annotation result (keyed by method + reference to allow coexistence)
    from .tools.annotation import _build_annotation_suffix

    cache_key = f"annotation_{_build_annotation_suffix(resolved_params.method, resolved_params.reference_data_id)}"
    await data_manager.save_result(data_id, cache_key, result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def analyze_spatial_statistics(
    data_id: str,
    params: Optional[SpatialStatisticsParameters] = None,
    context: Optional[Context] = None,
) -> SpatialStatisticsResult:
    """Analyze spatial statistics and autocorrelation patterns.

    Args:
        data_id: Dataset ID
        params: Analysis parameters (analysis_type, cluster_key, genes). See SpatialStatisticsParameters for all types.
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import spatial_statistics (squidpy is slow to import)
    from .tools.spatial_statistics import (
        analyze_spatial_statistics as _analyze_spatial_statistics,
    )

    resolved_params = _resolve_params(params, SpatialStatisticsParameters)

    # Call spatial statistics analysis function with ToolContext
    result = await _analyze_spatial_statistics(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save spatial statistics result (keyed by analysis_type to allow coexistence)
    cache_key = f"spatial_statistics_{resolved_params.analysis_type}"
    await data_manager.save_result(data_id, cache_key, result)

    # Note: Visualization should be created separately using create_visualization tool
    # This maintains clean separation between analysis and visualization

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def find_markers(
    data_id: str,
    params: DifferentialExpressionParameters,
    context: Optional[Context] = None,
) -> DifferentialExpressionResult:
    """Find differentially expressed genes between groups.

    Args:
        data_id: Dataset ID
        params: Required - group_key and optional method, group1/group2, n_top_genes, etc.
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    from .tools.differential import differential_expression

    from .tools.differential import _build_de_key

    result = await differential_expression(data_id, ctx, params)
    cache_key = _build_de_key(params.method, params.group1, params.group2)
    await data_manager.save_result(data_id, cache_key, result)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def compare_conditions(
    data_id: str,
    params: ConditionComparisonParameters,
    context: Optional[Context] = None,
) -> ConditionComparisonResult:
    """Compare experimental conditions using pseudobulk differential expression (DESeq2).

    Args:
        data_id: Dataset ID
        params: Required - condition_key, condition1, condition2, sample_key, etc.
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    from .tools.condition_comparison import compare_conditions as _compare_conditions

    result = await _compare_conditions(data_id, ctx, params)
    # Use a parametric key so multiple comparisons on the same dataset
    # don't silently overwrite each other.
    cache_key = f"condition_comparison_{params.condition1}_vs_{params.condition2}"
    await data_manager.save_result(data_id, cache_key, result)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def analyze_cnv(
    data_id: str,
    params: CNVParameters,
    context: Optional[Context] = None,
) -> CNVResult:
    """Analyze copy number variations (CNVs) in spatial transcriptomics data.

    Args:
        data_id: Dataset identifier
        params: Required - reference_key, reference_categories, and optional method/thresholds.
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    from .tools.cnv_analysis import _build_cnv_key, infer_cnv

    result = await infer_cnv(data_id=data_id, ctx=ctx, params=params)
    cache_key = _build_cnv_key(params)
    await data_manager.save_result(data_id, cache_key, result)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def analyze_velocity_data(
    data_id: str,
    params: Optional[RNAVelocityParameters] = None,
    context: Optional[Context] = None,
) -> RNAVelocityResult:
    """Analyze RNA velocity to understand cellular dynamics. Requires 'spliced' and 'unspliced' layers.

    Args:
        data_id: Dataset ID
        params: Velocity parameters (method, scvelo_mode, etc.)
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import velocity analysis tool
    from .tools.velocity import _build_velocity_key, analyze_rna_velocity

    resolved_params = _resolve_params(params, RNAVelocityParameters)

    # Call RNA velocity function with ToolContext
    result = await analyze_rna_velocity(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save velocity result (keyed by method+params to allow coexistence)
    cache_key = _build_velocity_key(resolved_params)
    await data_manager.save_result(data_id, cache_key, result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def analyze_trajectory_data(
    data_id: str,
    params: Optional[TrajectoryParameters] = None,
    context: Optional[Context] = None,
) -> TrajectoryResult:
    """Infer cellular trajectories and pseudotime ordering.

    Args:
        data_id: Dataset ID
        params: Trajectory parameters (method, root_cell, spatial_weight, etc.)
    """
    # Create ToolContext
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import trajectory function
    from .tools.trajectory import _build_trajectory_key, analyze_trajectory

    resolved_params = _resolve_params(params, TrajectoryParameters)

    # Call trajectory function
    result = await analyze_trajectory(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save trajectory result (keyed by method+params to allow coexistence)
    cache_key = _build_trajectory_key(resolved_params)
    await data_manager.save_result(data_id, cache_key, result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def integrate_samples(
    data_ids: list[str],
    params: Optional[IntegrationParameters] = None,
    context: Optional[Context] = None,
) -> IntegrationResult:
    """Integrate multiple spatial transcriptomics samples into a unified dataset.

    Args:
        data_ids: List of dataset IDs to integrate
        params: Integration parameters (method, batch_key, n_pcs, etc.)
    """
    # Create ToolContext for clean data access
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import to avoid slow startup
    from .tools.integration import integrate_samples as integrate_func

    resolved_params = _resolve_params(params, IntegrationParameters)

    # Call integration function with ToolContext
    # Note: integrate_func uses ctx.add_dataset() to store the integrated dataset
    result = await integrate_func(data_ids, ctx, resolved_params)

    # Save integration result
    integrated_id = result.data_id
    await data_manager.save_result(integrated_id, "integration", result)

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def deconvolve_data(
    data_id: str,
    params: DeconvolutionParameters,  # No default - LLM must provide parameters
    context: Optional[Context] = None,
) -> DeconvolutionResult:
    """Deconvolve spatial spots to estimate cell type proportions.

    Args:
        data_id: Dataset ID
        params: Required - method, cell_type_key, reference_data_id. See DeconvolutionParameters for all methods and options.
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import deconvolution tool
    from .tools.deconvolution import _build_deconvolution_key, deconvolve_spatial_data

    # Call deconvolution function with ToolContext
    result = await deconvolve_spatial_data(data_id, ctx, params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save deconvolution result (keyed by method+ref to allow coexistence)
    cache_key = _build_deconvolution_key(params.method, params.reference_data_id)
    await data_manager.save_result(data_id, cache_key, result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def identify_spatial_domains(
    data_id: str,
    params: Optional[SpatialDomainParameters] = None,
    context: Optional[Context] = None,
) -> SpatialDomainResult:
    """Identify spatial domains and tissue architecture.

    Args:
        data_id: Dataset ID
        params: Spatial domain parameters (method, n_domains, resolution, etc.)
    """
    # Create ToolContext for clean data access
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import to avoid slow startup
    from .tools.spatial_domains import identify_spatial_domains as identify_domains_func

    resolved_params = _resolve_params(params, SpatialDomainParameters)

    # Call spatial domains function with ToolContext
    result = await identify_domains_func(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save spatial domains result (keyed by method + params for coexistence)
    from .tools.spatial_domains import _build_domain_suffix

    cache_key = f"spatial_domains_{_build_domain_suffix(resolved_params.method, resolved_params.resolution, resolved_params.n_domains)}"
    await data_manager.save_result(data_id, cache_key, result)

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def analyze_cell_communication(
    data_id: str,
    params: CellCommunicationParameters,  # No default - LLM must provide parameters
    context: Optional[Context] = None,
) -> CellCommunicationResult:
    """Analyze cell-cell communication and ligand-receptor interaction patterns.

    Args:
        data_id: Dataset ID
        params: Required - species, cell_type_key, and method. For mouse with liana, set liana_resource='mouseconsensus'.
    """
    # Create ToolContext for clean data access
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import to avoid slow startup
    from .tools.cell_communication import (
        analyze_cell_communication as analyze_comm_func,
    )

    # Call cell communication function with ToolContext
    result = await analyze_comm_func(data_id, ctx, params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save communication result (keyed by method to allow coexistence)
    cache_key = f"cell_communication_{params.method}"
    await data_manager.save_result(data_id, cache_key, result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def analyze_enrichment(
    data_id: str,
    params: Optional[EnrichmentParameters] = None,
    context: Optional[Context] = None,
) -> EnrichmentResult:
    """Perform gene set enrichment analysis.

    Args:
        data_id: Dataset ID
        params: Required - species must be specified. See EnrichmentParameters for methods and gene_set_database options.
    """
    from .tools.enrichment import analyze_enrichment as analyze_enrichment_func

    # Create ToolContext
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Use default parameters if not provided (species is required by analyze_enrichment_func)
    if params is None:
        raise ParameterError(
            "EnrichmentParameters is required. Please specify at least 'species' parameter."
        )

    # Call enrichment analysis (all business logic is in tools/enrichment.py)
    result = await analyze_enrichment_func(data_id, ctx, params)

    # Save result (keyed by method + database to allow coexistence)
    from .tools.enrichment import _build_enrichment_key

    cache_key = _build_enrichment_key(params.method, params.gene_set_database)
    await data_manager.save_result(data_id, cache_key, result)

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def find_spatial_genes(
    data_id: str,
    params: Optional[SpatialVariableGenesParameters] = None,
    context: Optional[Context] = None,
) -> SpatialVariableGenesResult:
    """Identify spatially variable genes.

    Args:
        data_id: Dataset ID
        params: Spatial variable gene parameters (method, n_top_genes, etc.)
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import spatial genes tool
    from .tools.spatial_genes import identify_spatial_genes

    resolved_params = _resolve_params(params, SpatialVariableGenesParameters)

    # Call spatial genes function with ToolContext
    result = await identify_spatial_genes(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save spatial genes result (keyed by method to allow coexistence)
    cache_key = f"spatial_genes_{resolved_params.method}"
    await data_manager.save_result(data_id, cache_key, result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
@mcp_tool_error_handler()
async def register_spatial_data(
    source_id: str,
    target_id: str,
    params: Optional[RegistrationParameters] = None,
    context: Optional[Context] = None,
) -> dict[str, Any]:
    """Register/align spatial transcriptomics data across sections

    Args:
        source_id: Source dataset ID
        target_id: Target dataset ID to align to
        params: Registration parameters (method, alignment settings, etc.)

    Returns:
        Registration result with method, dataset IDs, spot counts, and registered spatial key
    """
    # Create ToolContext for unified data access
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import to avoid slow startup
    from .tools.spatial_registration import register_spatial_slices_mcp

    resolved_params = _resolve_params(params, RegistrationParameters)

    # Call registration function using ToolContext
    # Note: registration modifies adata in-place, changes reflected via reference
    result = await register_spatial_slices_mcp(
        source_id, target_id, ctx, resolved_params
    )

    # Save registration result for both datasets (registration is bilateral)
    await data_manager.save_result(source_id, "registration", result)
    await data_manager.save_result(target_id, "registration", result)

    return result


# ============== Data Export/Reload Tools ==============


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def export_data(
    data_id: str,
    path: Optional[str] = None,
    context: Optional[Context] = None,
) -> str:
    """Export dataset to disk for external script access.

    Args:
        data_id: Dataset ID to export
        path: Custom path (default: ~/.chatspatial/active/{data_id}.h5ad)

    Returns:
        Absolute path where data was exported
    """
    from pathlib import Path as PathLib

    from .utils.persistence import export_adata

    if context:
        await context.info(f"Exporting dataset '{data_id}'...")

    # Get dataset info
    dataset_info = await data_manager.get_dataset(data_id)
    adata = dataset_info["adata"]

    try:
        export_path = export_adata(data_id, adata, PathLib(path) if path else None)
        absolute_path = export_path.resolve()

        if context:
            await context.info(f"Dataset exported to: {absolute_path}")

        return f"Dataset '{data_id}' exported to: {absolute_path}"

    except Exception as e:
        error_msg = f"Failed to export dataset: {e}"
        if context:
            await context.error(error_msg)
        raise


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
@mcp_tool_error_handler()
async def reload_data(
    data_id: str,
    path: Optional[str] = None,
    context: Optional[Context] = None,
) -> str:
    """Reload dataset from disk after external script modifications.

    Args:
        data_id: Dataset ID to reload (must exist in MCP memory)
        path: Custom path (default: ~/.chatspatial/active/{data_id}.h5ad)

    Returns:
        Summary of reloaded dataset
    """
    from pathlib import Path as PathLib

    from .utils.persistence import load_adata_from_active

    if context:
        await context.info(f"Reloading dataset '{data_id}'...")

    try:
        adata = load_adata_from_active(data_id, PathLib(path) if path else None)

        # Update the in-memory dataset
        await data_manager.update_adata(data_id, adata)

        if context:
            await context.info(f"Dataset '{data_id}' reloaded successfully")

        return (
            f"Dataset '{data_id}' reloaded: "
            f"{adata.n_obs} cells × {adata.n_vars} genes"
        )

    except FileNotFoundError as e:
        error_msg = str(e)
        if context:
            await context.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Failed to reload dataset: {e}"
        if context:
            await context.error(error_msg)
        raise


# CLI entry point is in __main__.py (single source of truth)
# Use: python -m chatspatial server [options]
