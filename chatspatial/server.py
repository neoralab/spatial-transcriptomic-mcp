"""
Main server implementation for ChatSpatial using the Spatial MCP Adapter.
"""

from typing import Any, Optional, TypeVar

from mcp.server.fastmcp import Context

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
from .spatial_mcp_adapter import get_tool_annotations  # noqa: E402
from .utils.exceptions import ParameterError  # noqa: E402
from .utils.mcp_utils import mcp_tool_error_handler  # noqa: E402

# Create MCP server and adapter
mcp, adapter = create_spatial_mcp_server("ChatSpatial")

# Get data manager from adapter
data_manager = adapter.data_manager

P = TypeVar("P")


def _resolve_params(params: Optional[P], default_factory: type[P]) -> P:
    """Resolve optional params to a concrete model instance."""
    return params if params is not None else default_factory()


@mcp.tool(annotations=get_tool_annotations("load_data"))
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


@mcp.tool(annotations=get_tool_annotations("preprocess_data"))
@mcp_tool_error_handler()
async def preprocess_data(
    data_id: str,
    params: Optional[PreprocessingParameters] = None,
    context: Optional[Context] = None,
) -> PreprocessingResult:
    """Preprocess spatial transcriptomics data.

    Args:
        data_id: Dataset ID
        params: Preprocessing parameters

    Returns:
        PreprocessingResult with HVGs, PCA, clustering, and spatial neighbors
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


@mcp.tool(annotations=get_tool_annotations("compute_embeddings"))
@mcp_tool_error_handler()
async def compute_embeddings(
    data_id: str,
    params: Optional[EmbeddingParameters] = None,
    context: Optional[Context] = None,
) -> dict[str, Any]:
    """Compute dimensionality reduction, clustering, and neighbor graphs.

    Args:
        data_id: Dataset ID
        params: Embedding parameters (PCA, UMAP, clustering, etc.)

    Returns:
        Summary of computed embeddings
    """
    from .tools.embeddings import EmbeddingParameters
    from .tools.embeddings import compute_embeddings as compute_embeddings_func

    resolved = _resolve_params(params, EmbeddingParameters)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)
    result = await compute_embeddings_func(data_id, ctx, resolved)
    return result.model_dump()


@mcp.tool(annotations=get_tool_annotations("visualize_data"))
@mcp_tool_error_handler()
async def visualize_data(
    data_id: str,
    params: Optional[VisualizationParameters] = None,
    context: Optional[Context] = None,
) -> str:
    """Visualize spatial transcriptomics data.

    Args:
        data_id: Dataset ID
        params: Visualization parameters

    Plot types (11 unified types):
        - feature: Spatial/UMAP feature visualization (use basis='spatial'|'umap')
        - expression: Aggregated expression (subtype='heatmap'|'violin'|'dotplot'|'correlation')
        - deconvolution: Cell type proportions (subtype='spatial_multi'|'pie'|'dominant'|'diversity'|'umap'|'imputation')
        - communication: Cell-cell communication (subtype='dotplot'|'tileplot'|'circle_plot')
        - interaction: Spatial ligand-receptor pairs
        - trajectory: Pseudotime and fate analysis (subtype='pseudotime'|'circular'|'fate_map'|'gene_trends'|'fate_heatmap'|'palantir')
        - velocity: RNA velocity visualization (subtype='stream'|'phase'|'proportions'|'heatmap'|'paga')
        - statistics: Spatial statistics (subtype='neighborhood'|'co_occurrence'|'ripley'|'moran'|'centrality'|'getis_ord')
        - enrichment: Pathway enrichment (subtype='barplot'|'dotplot')
        - cnv: Copy number variation (subtype='heatmap'|'spatial')
        - integration: Batch integration quality (subtype='batch'|'cluster'|'highlight')

    Export options (in params):
        - output_path: Custom save path (default: ./visualizations/)
        - output_format: png, pdf, svg, eps, tiff (default: png)
        - dpi: Resolution (default: 300)

    Returns:
        Path to saved visualization file
    """
    from .tools.visualization import visualize_data as visualize_func

    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    resolved_params = _resolve_params(params, VisualizationParameters)
    result = await visualize_func(data_id, ctx, resolved_params)

    if result:
        return result
    else:
        return "Visualization generation failed, please check the data and parameter settings."


@mcp.tool(annotations=get_tool_annotations("annotate_cell_types"))
@mcp_tool_error_handler()
async def annotate_cell_types(
    data_id: str,
    params: Optional[AnnotationParameters] = None,
    context: Optional[Context] = None,
) -> AnnotationResult:
    """Annotate cell types in spatial transcriptomics data.

    Args:
        data_id: Dataset ID
        params: Annotation parameters

    Key requirements:
        - Reference methods (tangram, scanvi): reference_data_id must be preprocessed first
        - cell_type_key: Auto-detected if None

    Returns:
        AnnotationResult with cell type assignments and confidence scores
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import annotation tool (avoids slow startup)
    from .tools.annotation import annotate_cell_types

    resolved_params = _resolve_params(params, AnnotationParameters)

    # Call annotation function with ToolContext
    result = await annotate_cell_types(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save annotation result
    await data_manager.save_result(data_id, "annotation", result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(annotations=get_tool_annotations("analyze_spatial_statistics"))
@mcp_tool_error_handler()
async def analyze_spatial_statistics(
    data_id: str,
    params: Optional[SpatialStatisticsParameters] = None,
    context: Optional[Context] = None,
) -> SpatialStatisticsResult:
    """Analyze spatial statistics and autocorrelation patterns.

    Args:
        data_id: Dataset ID
        params: Analysis parameters (analysis_type, cluster_key, genes)

    Analysis types:
        - Gene-based: moran, local_moran, geary, getis_ord, bivariate_moran
        - Group-based (requires cluster_key): neighborhood, co_occurrence, ripley
        - Categorical: join_count (binary), local_join_count (multi-category)
        - Network: centrality, network_properties

    Returns:
        SpatialStatisticsResult with statistics and p-values
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

    # Save spatial statistics result
    await data_manager.save_result(data_id, "spatial_statistics", result)

    # Note: Visualization should be created separately using create_visualization tool
    # This maintains clean separation between analysis and visualization

    return result


@mcp.tool(annotations=get_tool_annotations("find_markers"))
@mcp_tool_error_handler()
async def find_markers(
    data_id: str,
    params: DifferentialExpressionParameters,
    context: Optional[Context] = None,
) -> DifferentialExpressionResult:
    """Find differentially expressed genes between groups.

    Args:
        data_id: Dataset ID
        params: Required - group_key and optional method, group1/group2, etc.

    Returns:
        DifferentialExpressionResult with top marker genes
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    from .tools.differential import differential_expression

    result = await differential_expression(data_id, ctx, params)
    await data_manager.save_result(data_id, "differential_expression", result)
    return result


@mcp.tool(annotations=get_tool_annotations("compare_conditions"))
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

    Returns:
        ConditionComparisonResult with differential expression results
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    from .tools.condition_comparison import compare_conditions as _compare_conditions

    result = await _compare_conditions(data_id, ctx, params)
    await data_manager.save_result(data_id, "condition_comparison", result)
    return result


@mcp.tool(annotations=get_tool_annotations("analyze_cnv"))
@mcp_tool_error_handler()
async def analyze_cnv(
    data_id: str,
    params: CNVParameters,
    context: Optional[Context] = None,
) -> CNVResult:
    """Analyze copy number variations (CNVs) in spatial transcriptomics data.

    Args:
        data_id: Dataset identifier
        params: Required - reference_key, reference_categories, and optional method/thresholds

    Returns:
        CNVResult with CNV scores and optional clone assignments
    """
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    from .tools.cnv_analysis import infer_cnv

    result = await infer_cnv(data_id=data_id, ctx=ctx, params=params)
    await data_manager.save_result(data_id, "cnv_analysis", result)
    return result


@mcp.tool(annotations=get_tool_annotations("analyze_velocity_data"))
@mcp_tool_error_handler()
async def analyze_velocity_data(
    data_id: str,
    params: Optional[RNAVelocityParameters] = None,
    context: Optional[Context] = None,
) -> RNAVelocityResult:
    """Analyze RNA velocity to understand cellular dynamics.

    Args:
        data_id: Dataset ID (must have 'spliced' and 'unspliced' layers)
        params: method='scvelo' (modes: deterministic/stochastic/dynamical) or 'velovi'

    Returns:
        RNAVelocityResult with velocity vectors and latent time
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import velocity analysis tool
    from .tools.velocity import analyze_rna_velocity

    resolved_params = _resolve_params(params, RNAVelocityParameters)

    # Call RNA velocity function with ToolContext
    result = await analyze_rna_velocity(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save velocity result
    await data_manager.save_result(data_id, "rna_velocity", result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(annotations=get_tool_annotations("analyze_trajectory_data"))
@mcp_tool_error_handler()
async def analyze_trajectory_data(
    data_id: str,
    params: Optional[TrajectoryParameters] = None,
    context: Optional[Context] = None,
) -> TrajectoryResult:
    """Infer cellular trajectories and pseudotime.

    Args:
        data_id: Dataset ID
        params: method='cellrank' (requires velocity), 'palantir', or 'dpt'

    Returns:
        TrajectoryResult with pseudotime and fate probabilities
    """
    # Create ToolContext
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import trajectory function
    from .tools.trajectory import analyze_trajectory

    resolved_params = _resolve_params(params, TrajectoryParameters)

    # Call trajectory function
    result = await analyze_trajectory(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save trajectory result
    await data_manager.save_result(data_id, "trajectory", result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(annotations=get_tool_annotations("integrate_samples"))
@mcp_tool_error_handler()
async def integrate_samples(
    data_ids: list[str],
    params: Optional[IntegrationParameters] = None,
    context: Optional[Context] = None,
) -> IntegrationResult:
    """Integrate multiple spatial transcriptomics samples.

    Args:
        data_ids: List of dataset IDs to integrate
        params: method='harmony' (default), 'bbknn', 'scanorama', or 'scvi'

    Returns:
        IntegrationResult with integrated dataset ID
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


@mcp.tool(annotations=get_tool_annotations("deconvolve_data"))
@mcp_tool_error_handler()
async def deconvolve_data(
    data_id: str,
    params: DeconvolutionParameters,  # No default - LLM must provide parameters
    context: Optional[Context] = None,
) -> DeconvolutionResult:
    """Deconvolve spatial spots to estimate cell type proportions.

    Args:
        data_id: Dataset ID
        params: Required - method, cell_type_key, reference_data_id

    Methods:
        - flashdeconv: Fast sketch-based method (default, recommended)
        - cell2location: Deep learning, accurate but slow (requires scvi-tools)
        - rctd: R-based, modes: doublet (high-res), full (Visium), multi
        - destvi, stereoscope, tangram: Alternative deep learning methods
        - spotlight, card: R-based methods (card supports spatial imputation)

    Returns:
        DeconvolutionResult with cell type proportions per spot
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import deconvolution tool
    from .tools.deconvolution import deconvolve_spatial_data

    # Call deconvolution function with ToolContext
    result = await deconvolve_spatial_data(data_id, ctx, params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save deconvolution result
    await data_manager.save_result(data_id, "deconvolution", result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(annotations=get_tool_annotations("identify_spatial_domains"))
@mcp_tool_error_handler()
async def identify_spatial_domains(
    data_id: str,
    params: Optional[SpatialDomainParameters] = None,
    context: Optional[Context] = None,
) -> SpatialDomainResult:
    """Identify spatial domains and tissue architecture.

    Args:
        data_id: Dataset ID
        params: method='spagcn' (default, uses histology), 'leiden', 'louvain', 'stagate', 'graphst'

    Returns:
        SpatialDomainResult with domain_key for visualization
    """
    # Create ToolContext for clean data access
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import to avoid slow startup
    from .tools.spatial_domains import identify_spatial_domains as identify_domains_func

    resolved_params = _resolve_params(params, SpatialDomainParameters)

    # Call spatial domains function with ToolContext
    result = await identify_domains_func(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save spatial domains result
    await data_manager.save_result(data_id, "spatial_domains", result)

    return result


@mcp.tool(annotations=get_tool_annotations("analyze_cell_communication"))
@mcp_tool_error_handler()
async def analyze_cell_communication(
    data_id: str,
    params: CellCommunicationParameters,  # No default - LLM must provide parameters
    context: Optional[Context] = None,
) -> CellCommunicationResult:
    """Analyze cell-cell communication patterns.

    Args:
        data_id: Dataset ID
        params: Required - species, cell_type_key, and method

    Methods:
        - fastccc: Fast C++ implementation (default). Human and mouse supported
        - liana: Multi-method consensus. Use liana_resource for database selection
        - cellphonedb: Statistical permutation-based analysis
        - cellchat_r: R-based CellChat (requires rpy2)

    Species configuration:
        - human: liana_resource="consensus" (default)
        - mouse: liana_resource="mouseconsensus"

    Returns:
        CellCommunicationResult with significant ligand-receptor interactions
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

    # Save communication result
    await data_manager.save_result(data_id, "cell_communication", result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(annotations=get_tool_annotations("analyze_enrichment"))
@mcp_tool_error_handler()
async def analyze_enrichment(
    data_id: str,
    params: Optional[EnrichmentParameters] = None,
    context: Optional[Context] = None,
) -> EnrichmentResult:
    """Perform gene set enrichment analysis.

    Args:
        data_id: Dataset ID
        params: Required - species must be specified ('human' or 'mouse')

    Methods:
        - pathway_ora: Over-representation analysis (default)
        - pathway_gsea: Gene Set Enrichment Analysis
        - spatial_enrichmap: Spatial enrichment mapping
        - pathway_enrichr, pathway_ssgsea: Alternative methods

    Databases (gene_set_database):
        - KEGG_Pathways (recommended), Reactome_Pathways, MSigDB_Hallmark
        - GO_Biological_Process (default), GO_Molecular_Function, GO_Cellular_Component

    Returns:
        EnrichmentResult with enriched pathways and statistics
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

    # Save result
    await data_manager.save_result(data_id, "enrichment", result)

    return result


@mcp.tool(annotations=get_tool_annotations("find_spatial_genes"))
@mcp_tool_error_handler()
async def find_spatial_genes(
    data_id: str,
    params: Optional[SpatialVariableGenesParameters] = None,
    context: Optional[Context] = None,
) -> SpatialVariableGenesResult:
    """Identify spatially variable genes.

    Args:
        data_id: Dataset ID
        params: method='sparkx' (default), 'flashs' (Python-native fast), or 'spatialde' (Gaussian process)

    Returns:
        SpatialVariableGenesResult with ranked genes and statistics
    """
    # Create ToolContext for clean data access (no redundant dict wrapping)
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import spatial genes tool
    from .tools.spatial_genes import identify_spatial_genes

    resolved_params = _resolve_params(params, SpatialVariableGenesParameters)

    # Call spatial genes function with ToolContext
    result = await identify_spatial_genes(data_id, ctx, resolved_params)

    # Note: No writeback needed - adata modifications are in-place on the same object

    # Save spatial genes result
    await data_manager.save_result(data_id, "spatial_genes", result)

    # Visualization should be done separately via visualization tools

    return result


@mcp.tool(annotations=get_tool_annotations("register_spatial_data"))
@mcp_tool_error_handler()
async def register_spatial_data(
    source_id: str,
    target_id: str,
    method: str = "paste",
    context: Optional[Context] = None,
) -> dict[str, Any]:
    """Register/align spatial transcriptomics data across sections

    Args:
        source_id: Source dataset ID
        target_id: Target dataset ID to align to
        method: Registration method (paste, stalign)

    Returns:
        Registration result with transformation matrix
    """
    # Create ToolContext for unified data access
    ctx = ToolContext(_data_manager=data_manager, _mcp_context=context)

    # Lazy import to avoid slow startup
    from .tools.spatial_registration import register_spatial_slices_mcp

    # Call registration function using ToolContext
    # Note: registration modifies adata in-place, changes reflected via reference
    result = await register_spatial_slices_mcp(source_id, target_id, ctx, method)

    # Save registration result
    await data_manager.save_result(source_id, "registration", result)

    return result


# ============== Data Export/Reload Tools ==============


@mcp.tool(annotations=get_tool_annotations("export_data"))
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


@mcp.tool(annotations=get_tool_annotations("reload_data"))
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
