"""
Entry point for ChatSpatial.

This module provides the command-line interface for starting the
ChatSpatial server using stdio, SSE, or Streamable HTTP transport.
"""

import os
import sys
import traceback
from typing import Literal, cast

import click
from mcp.server.transport_security import TransportSecuritySettings

# Initialize runtime configuration (SSOT - all config in one place)
# This import triggers init_runtime() which configures:
# - Environment variables (TQDM_DISABLE, DASK_*)
# - Warning filters
# - Scanpy settings
from . import config  # noqa: F401
from .server import mcp


@click.group()
def cli():
    """ChatSpatial - AI-powered spatial transcriptomics analysis"""
    pass


@cli.command()
@click.option("--port", default=8000, help="Port to listen on for HTTP transports")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    help="Transport type (stdio, sse, or streamable-http)",
)
@click.option(
    "--host",
    default="127.0.0.1",  # nosec B104 - Default to localhost for security
    help="Host to bind to for HTTP transports",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print initialization info",
)
@click.option(
    "--cloud-run",
    is_flag=True,
    default=False,
    help="Cloud Run mode: use streamable-http on 0.0.0.0:$PORT",
)
def server(
    port: int,
    transport: str,
    host: str,
    log_level: str,
    verbose: bool,
    cloud_run: bool,
):
    """Start the ChatSpatial server.

    This command starts the ChatSpatial server using stdio, SSE, or Streamable HTTP transport.
    For stdio transport, the server communicates through standard input/output.
    For SSE transport, the server starts an HTTP server on the specified host and port.
    """
    try:
        if verbose:
            # Re-initialize with verbose output
            config.init_runtime(verbose=True)

        # Cloud Run requires HTTP server bound to all interfaces and PORT env var
        if cloud_run:
            host = "0.0.0.0"
            transport = "streamable-http"
            port = int(os.environ.get("PORT", str(port)))

        print(
            f"Starting ChatSpatial server with {transport} transport on {host}:{port}...",
            file=sys.stderr,
        )

        # Set server settings
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.settings.log_level = cast(
            Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], log_level
        )

        # When binding to 0.0.0.0 or a non-localhost address (e.g. behind a
        # reverse proxy / Cloudflare Tunnel), the FastMCP constructor has
        # already set transport_security with DNS-rebinding protection locked
        # to localhost-only allowed_hosts because the default host is
        # "127.0.0.1".  Updating mcp.settings.host above does NOT retroactively
        # change that security policy, so requests arriving with the cloudflared
        # domain in the Host header are rejected with "Invalid Host header".
        # Fix: disable DNS rebinding protection for non-localhost deployments.
        _localhost_hosts = ("127.0.0.1", "localhost", "::1")
        if host not in _localhost_hosts:
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            )

        # Run the server with the specified transport
        mcp.run(transport=cast(Literal["stdio", "sse", "streamable-http"], transport))

    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for ChatSpatial CLI"""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
