<div align="center">

# ChatSpatial

**MCP server for spatial transcriptomics analysis via natural language**

[![CI](https://github.com/cafferychen777/ChatSpatial/actions/workflows/ci.yml/badge.svg)](https://github.com/cafferychen777/ChatSpatial/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chatspatial)](https://pypi.org/project/chatspatial/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-available-blue)](https://cafferychen777.github.io/ChatSpatial/)

</div>

---

<table>
<tr>
<td width="50%">

### Before
```python
import scanpy as sc
import squidpy as sq
adata = sc.read_h5ad("data.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
# ... 40 more lines
```

</td>
<td width="50%">

### After
```text
"Load my Visium data and identify
 spatial domains"
```

```
Loaded 3,456 spots, 18,078 genes
Identified 7 spatial domains
Generated visualization
```

</td>
</tr>
</table>

---

## Quick Start

**Using Claude Code / Codex / OpenCode?** Just paste this:
```
Install ChatSpatial following https://github.com/cafferychen777/ChatSpatial/blob/main/INSTALLATION.md
```

<details>
<summary>Manual installation</summary>

```bash
# Install uv (recommended - handles complex dependencies)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
python3 -m venv venv && source venv/bin/activate
uv pip install chatspatial

# Configure (use your venv Python path)
claude mcp add chatspatial /path/to/venv/bin/python -- -m chatspatial server
```
</details>

> **Works with any MCP-compatible client** — not just Claude. Use with [OpenCode](https://opencode.ai/), Codex, or any client supporting [Model Context Protocol](https://modelcontextprotocol.io/). Configure your preferred LLM (Qwen, DeepSeek, Doubao, etc.) as the backend.

See [Installation Guide](INSTALLATION.md) for detailed setup including virtual environments and all MCP clients.

Need remote/shared deployment? Use [Cloud Run Deployment Guide](docs/advanced/cloud-run.md) to run ChatSpatial as an HTTPS MCP endpoint.

---

## Use

```text
Load /path/to/spatial_data.h5ad and show me the tissue structure
```

```text
Identify spatial domains using SpaGCN
```

```text
Find spatially variable genes and create a heatmap
```

---

## Capabilities

| Category | Methods |
|----------|---------|
| **Spatial Domains** | SpaGCN, STAGATE, GraphST, Leiden, Louvain |
| **Deconvolution** | FlashDeconv, Cell2location, RCTD, DestVI, Stereoscope, SPOTlight, Tangram, CARD |
| **Cell Communication** | LIANA+, CellPhoneDB, CellChat, FastCCC |
| **Cell Type Annotation** | Tangram, scANVI, CellAssign, mLLMCelltype, scType, SingleR |
| **Trajectory & Velocity** | CellRank, Palantir, DPT, scVelo, VeloVI |
| **Spatial Statistics** | Moran's I, Local Moran, Geary's C, Getis-Ord Gi*, Ripley's K, Neighborhood Enrichment |
| **Enrichment** | GSEA, ORA, Enrichr, ssGSEA, Spatial EnrichMap |
| **Spatial Genes** | SpatialDE, SPARK-X |
| **Integration** | Harmony, BBKNN, Scanorama, scVI |
| **Other** | CNV Analysis, Spatial Registration |

**60+ methods** across 15 categories. **Supports** 10x Visium, Xenium, Slide-seq v2, MERFISH, seqFISH.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](INSTALLATION.md) | Virtual environment setup, all platforms |
| [Quick Start](docs/quickstart.md) | 5-minute first analysis |
| [Examples](docs/examples.md) | Step-by-step workflows |
| [Methods Reference](docs/advanced/methods-reference.md) | All 20 tools with parameters |
| [Full Docs](https://cafferychen777.github.io/ChatSpatial/) | Complete reference |

---

## Citation

If you use ChatSpatial in your research, please cite:

```bibtex
@article{Yang2026.02.26.708361,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {ChatSpatial: Schema-Enforced Agentic Orchestration for Reproducible and Cross-Platform Spatial Transcriptomics},
  elocation-id = {2026.02.26.708361},
  year = {2026},
  doi = {10.64898/2026.02.26.708361},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2026/03/01/2026.02.26.708361},
  journal = {bioRxiv}
}
```

<div align="center">

**MIT License** · [GitHub](https://github.com/cafferychen777/ChatSpatial) · [Issues](https://github.com/cafferychen777/ChatSpatial/issues)

</div>

<!-- mcp-name: io.github.cafferychen777/chatspatial -->
