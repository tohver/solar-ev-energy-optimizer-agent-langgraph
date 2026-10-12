"""Command-line interface: ``energy-advisor <command>``. Replaces the former Jupyter notebooks."""

import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from energy_advisor.config import MissingApiKeyError, Settings, get_settings
from energy_advisor.logging_config import configure_logging

app = typer.Typer(help="Solar & EV energy optimizer agent.", no_args_is_help=True, add_completion=False)
console = Console()


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level)
    return settings


def handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Turn expected user errors into a short message and exit code 1 instead of a traceback."""
    from energy_advisor.rag.ingest import IndexExistsError
    from energy_advisor.rag.knowledge_base import IndexNotBuiltError

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (MissingApiKeyError, IndexNotBuiltError, IndexExistsError, FileNotFoundError) as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(code=1) from exc

    return wrapper


# ----------------------------------------------------------------------------- data
@app.command("seed-db")
@handle_errors
def seed_db(
    days: Annotated[int, typer.Option(min=1, help="Number of days of history, ending yesterday.")] = 30,
    seed: Annotated[int, typer.Option(help="Random seed; the same seed gives the same data.")] = 42,
) -> None:
    """Create the SQLite database with reproducible sample usage and solar data."""
    from energy_advisor.data.repository import EnergyRepository
    from energy_advisor.data.seed import seed_database

    settings = _settings()
    result = seed_database(EnergyRepository(settings.paths.db), days=days, seed=seed)
    console.print(
        f"[green]Database ready:[/] {settings.paths.db}\n"
        f"  {result.first_day} .. {result.last_day}: {result.usage_rows} usage rows, "
        f"{result.generation_rows} solar rows (replaced {sum(result.deleted.values())} old rows)"
    )


# ----------------------------------------------------------------------------- RAG
@app.command("build-index")
@handle_errors
def build_index_cmd(
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Replace an existing index.")] = False,
) -> None:
    """Embed the knowledge-base documents into the Chroma vector store."""
    from energy_advisor.llm import make_embeddings
    from energy_advisor.rag.ingest import build_index

    settings = _settings()
    with console.status("Embedding documents..."):
        result = build_index(
            settings.paths.knowledge_base,
            settings.paths.vectorstore,
            make_embeddings(settings),
            settings.rag,
            rebuild,
        )
    console.print(
        f"[green]Index ready:[/] {result.documents} documents, {result.chunks} chunks in {result.persist_dir}"
    )


@app.command("rag-benchmark")
@handle_errors
def rag_benchmark(k: Annotated[int, typer.Option(min=1, help="Cut-off for hit@k.")] = 3) -> None:
    """Compare vector, hybrid and hybrid + re-ranking retrieval on labelled queries (hit@k, MRR)."""
    from energy_advisor.rag.benchmark import METHODS, run_benchmark
    from energy_advisor.services.container import build_services

    services = build_services(_settings())
    with console.status("Running retrieval benchmark (makes LLM calls)..."):
        rows, summary = run_benchmark(services.knowledge_base, k=k)

    detail = Table(title="Rank of the expected document (- = not in top k)")
    detail.add_column("query")
    for m in METHODS:
        detail.add_column(m, justify="right")
    for row in rows:
        detail.add_row(str(row["query"]), *[str(row[m] or "-") for m in METHODS])
    console.print(detail)

    table = Table(title=f"Retrieval quality (k={k})")
    for col in ("method", f"hit@{k}", "MRR"):
        table.add_column(col)
    for s in summary:
        table.add_row(s.method, f"{s.hit_at_k:.2f}", f"{s.mrr:.2f}")
    console.print(table)


# ----------------------------------------------------------------------------- agent
@app.command()
@handle_errors
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in quotes.")],
    context: Annotated[
        str | None, typer.Option(help="Extra context; defaults to the configured location.")
    ] = None,
    show_tools: Annotated[bool, typer.Option(help="Print the tool calls the agent made.")] = True,
) -> None:
    """Ask the energy advisor a question."""
    from energy_advisor.agent.graph import EnergyAdvisor
    from energy_advisor.agent.trace import final_answer, tool_calls

    settings = _settings()
    advisor = EnergyAdvisor.from_settings(settings)
    started = time.perf_counter()
    with console.status("Thinking..."):
        result = advisor.invoke(question, context=context)
    elapsed = time.perf_counter() - started

    if show_tools:
        table = Table(title=f"Tool calls ({elapsed:.1f} s total)", show_lines=False)
        table.add_column("tool", style="cyan")
        table.add_column("arguments")
        for call in tool_calls(result):
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            table.add_row(call["name"], args)
        console.print(table)
    console.print(Panel(Markdown(final_answer(result)), title="Answer", border_style="green"))


@app.command("eval")
@handle_errors
def evaluate(
    cases_file: Annotated[
        Path | None, typer.Option("--cases", help="YAML file with cases (default: bundled).")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Run only the first N cases.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output folder (default: reports/).")] = None,
) -> None:
    """Run the evaluation cases, score them (LLM judge + tool usage) and write a report."""
    from energy_advisor.agent.graph import EnergyAdvisor
    from energy_advisor.evaluation.cases import load_cases
    from energy_advisor.evaluation.judge import LLMJudge
    from energy_advisor.evaluation.report import ALL_METRICS, generate_report, render_markdown, save_report
    from energy_advisor.evaluation.runner import CaseResult, run_evaluation
    from energy_advisor.llm import make_chat_model
    from energy_advisor.services.container import build_services

    settings = _settings()
    services = build_services(settings)
    advisor = EnergyAdvisor.from_settings(settings, services)
    judge = LLMJudge.from_llm(make_chat_model(settings, settings.llm.judge_model))
    cases = load_cases(cases_file)[:limit]

    def progress(r: CaseResult) -> None:
        console.print(
            f"  {r.test_id:<16} response={r.response_eval.overall:.2f}  "
            f"tools(appropriateness={r.tool_eval.appropriateness:.2f}, completeness={r.tool_eval.completeness:.2f})"
        )

    console.print(f"Running {len(cases)} cases...")
    results = run_evaluation(advisor, judge, cases, now=services.now, on_result=progress)
    report = generate_report(results, now=services.now())
    out_dir = out or settings.paths.reports
    paths = save_report(report, results, out_dir)

    try:
        from energy_advisor.viz.charts import save_evaluation_heatmap

        rows = [row.model_dump() for row in report.per_test]
        paths["heatmap"] = save_evaluation_heatmap(rows, ALL_METRICS, paths["markdown"].with_suffix(".png"))
    except ImportError:
        console.print("[yellow]Install the 'viz' extra for the heatmap: uv sync --extra viz[/]")

    console.print(Markdown(render_markdown(report)))
    for kind, path in paths.items():
        console.print(f"[green]{kind}:[/] {path}")


@app.command()
@handle_errors
def report(
    out: Annotated[Path | None, typer.Option(help="Output folder (default: reports/charts).")] = None,
) -> None:
    """Generate the charts (usage profile, usage vs. solar, tomorrow's window, savings) as PNG files."""
    try:
        from energy_advisor.viz.charts import generate_charts
    except ImportError as exc:
        console.print("[red]Charts need the 'viz' extra:[/] uv sync --extra viz")
        raise typer.Exit(code=1) from exc
    from energy_advisor.services.container import build_services

    settings = _settings()
    for path in generate_charts(build_services(settings), out or settings.paths.reports / "charts"):
        console.print(f"[green]chart:[/] {path}")


if __name__ == "__main__":
    app()
