"""Entry point for the ``midas`` CLI.

Subcommand groups are stubbed out and will be implemented in subsequent
slices (sources, extractors, graph). Keeping the surface stable now lets
us depend on it from docs and tests.
"""

from __future__ import annotations

import typer

from midas import __version__

app = typer.Typer(
    name="midas",
    help="Map cash flow between companies in a sector.",
    no_args_is_help=True,
    add_completion=False,
)

ingest_app = typer.Typer(help="Fetch raw documents from sources.", no_args_is_help=True)
graph_app = typer.Typer(help="Build and render the cash-flow graph.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(graph_app, name="graph")


@app.command()
def version() -> None:
    """Print the installed midas version."""
    typer.echo(__version__)


@app.command()
def init() -> None:
    """Create the database schema and seed the entity registry."""
    typer.echo("not implemented yet")
    raise typer.Exit(code=1)


@ingest_app.command("sec")
def ingest_sec(
    ticker: str = typer.Option(..., help="Ticker symbol, e.g. MSFT."),
) -> None:
    """Fetch SEC EDGAR filings for one ticker."""
    typer.echo(f"not implemented yet (would fetch SEC filings for {ticker})")
    raise typer.Exit(code=1)


@graph_app.command("render")
def graph_render(
    sector: str = typer.Option(..., help="Sector tag, e.g. ai."),
) -> None:
    """Render the cash-flow graph for a sector to interactive HTML."""
    typer.echo(f"not implemented yet (would render graph for sector {sector})")
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
