"""CLI 인터페이스

Click 기반 커맨드라인 인터페이스를 제공합니다.

사용법:
    airgap-kor-search index ./documents/
    airgap-kor-search search "한국어 형태소 분석"
    airgap-kor-search list
    airgap-kor-search stats
    airgap-kor-search serve
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from airgap_kor_search import __version__

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


def get_engine(config_path: Optional[str] = None):
    """SearchEngine 인스턴스를 생성합니다."""
    from airgap_kor_search.searcher import SearchEngine

    return SearchEngine.from_config_path(config_path)


# ── 메인 그룹 ─────────────────────────────────────────────


@click.group()
@click.version_option(version=__version__, prog_name="airgap-kor-search")
@click.option(
    "-c", "--config",
    type=click.Path(),
    default=None,
    help="설정 파일 경로 (기본: ./airgap_data/config.json)",
)
@click.option("-v", "--verbose", is_flag=True, help="상세 로그 출력")
@click.pass_context
def main(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """🔍 에어갭 환경을 위한 경량 한국어 문서 검색 엔진"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


# ── index 커맨드 ──────────────────────────────────────────


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--no-recursive",
    is_flag=True,
    default=False,
    help="하위 디렉토리를 탐색하지 않음",
)
@click.pass_context
def index(ctx: click.Context, path: str, no_recursive: bool) -> None:
    """문서를 인덱싱합니다.

    PATH는 파일 또는 디렉토리 경로입니다.
    """
    config_path = ctx.obj["config_path"]
    target = Path(path)

    with console.status("[bold green]인덱싱 준비 중..."):
        engine = get_engine(config_path)
        engine.open()

    try:
        if target.is_file():
            console.print(f"📄 파일 인덱싱: [cyan]{target}[/cyan]")
            with console.status("[bold green]인덱싱 중..."):
                result = engine.index_file(target)
        else:
            recursive = not no_recursive
            file_count = sum(
                1 for f in target.rglob("*") if f.is_file()
            ) if recursive else sum(
                1 for f in target.iterdir() if f.is_file()
            )
            console.print(
                f"📂 디렉토리 인덱싱: [cyan]{target}[/cyan] "
                f"(약 {file_count}개 파일, 재귀={'예' if recursive else '아니오'})"
            )
            with console.status("[bold green]인덱싱 중..."):
                result = engine.index_directory(target, recursive=recursive)

        # 결과 출력
        _print_indexing_result(result)

        if result.errors:
            console.print("\n[yellow]⚠️ 경고:[/yellow]")
            for err in result.errors:
                console.print(f"  • {err}")

    finally:
        engine.close()


def _print_indexing_result(result) -> None:
    """인덱싱 결과를 출력합니다."""
    table = Table(title="인덱싱 결과", show_header=False, border_style="green")
    table.add_column("항목", style="bold")
    table.add_column("값", justify="right")

    table.add_row("처리된 문서", f"{result.documents_processed}개")
    table.add_row("생성된 청크", f"{result.chunks_created}개")
    table.add_row("소요 시간", f"{result.elapsed_sec:.2f}초")

    console.print(table)


# ── search 커맨드 ─────────────────────────────────────────


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=None, help="결과 수 (기본: 5)")
@click.option(
    "-t", "--threshold",
    type=float,
    default=None,
    help="최소 유사도 점수 (0.0~1.0)",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    top_k: Optional[int],
    threshold: Optional[float],
) -> None:
    """쿼리로 문서를 검색합니다.

    QUERY는 검색할 한국어 텍스트입니다.
    """
    config_path = ctx.obj["config_path"]

    with console.status("[bold green]검색 엔진 로딩 중..."):
        engine = get_engine(config_path)
        engine.open()

    try:
        with console.status(f"[bold green]검색 중: '{query}'"):
            response = engine.search(
                query, top_k=top_k, score_threshold=threshold
            )

        _print_search_response(response)

    finally:
        engine.close()


def _print_search_response(response) -> None:
    """검색 결과를 출력합니다."""
    header = Text()
    header.append(f"🔍 '{response.query}'", style="bold")
    header.append(f" → {response.total_found}건", style="green")
    header.append(f" ({response.elapsed_ms:.1f}ms)", style="dim")
    console.print(header)
    console.print()

    if not response.results:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        console.print("💡 다른 키워드로 시도하거나, 문서를 먼저 인덱싱해주세요.")
        return

    for i, result in enumerate(response.results, 1):
        # 점수에 따른 색상
        if result.score >= 0.8:
            score_style = "bold green"
        elif result.score >= 0.6:
            score_style = "yellow"
        else:
            score_style = "dim"

        # 텍스트 미리보기 (최대 200자)
        preview = result.text[:200]
        if len(result.text) > 200:
            preview += "..."

        panel = Panel(
            preview,
            title=f"[{score_style}]#{i} ({result.score_percent}%)[/{score_style}]",
            subtitle=f"[dim]{result.doc_path} (청크 #{result.chunk_index})[/dim]",
            border_style=score_style,
            padding=(0, 1),
        )
        console.print(panel)


# ── list 커맨드 ───────────────────────────────────────────


@main.command(name="list")
@click.pass_context
def list_docs(ctx: click.Context) -> None:
    """인덱싱된 문서 목록을 출력합니다."""
    config_path = ctx.obj["config_path"]

    engine = get_engine(config_path)
    engine.open()

    try:
        docs = engine.list_documents()

        if not docs:
            console.print("[yellow]인덱싱된 문서가 없습니다.[/yellow]")
            console.print("💡 [cyan]airgap-kor-search index <경로>[/cyan]로 문서를 추가하세요.")
            return

        table = Table(title=f"인덱싱된 문서 ({len(docs)}개)")
        table.add_column("#", style="dim", justify="right")
        table.add_column("문서 경로", style="cyan")
        table.add_column("청크 수", justify="right")

        for i, doc in enumerate(docs, 1):
            table.add_row(str(i), doc["doc_path"], f"{doc['chunk_count']}개")

        console.print(table)

    finally:
        engine.close()


# ── delete 커맨드 ─────────────────────────────────────────


@main.command()
@click.argument("doc_path")
@click.option("-y", "--yes", is_flag=True, help="확인 없이 삭제")
@click.pass_context
def delete(ctx: click.Context, doc_path: str, yes: bool) -> None:
    """인덱스에서 문서를 삭제합니다.

    DOC_PATH는 인덱싱 시 사용한 문서 경로입니다.
    """
    config_path = ctx.obj["config_path"]

    engine = get_engine(config_path)
    engine.open()

    try:
        if not yes:
            if not click.confirm(f"'{doc_path}'을(를) 인덱스에서 삭제하시겠습니까?"):
                console.print("[dim]취소되었습니다.[/dim]")
                return

        deleted = engine.delete_document(doc_path)

        if deleted > 0:
            console.print(
                f"[green]✅ 삭제 완료:[/green] {doc_path} ({deleted}개 청크)"
            )
        else:
            console.print(f"[yellow]해당 문서를 찾을 수 없습니다: {doc_path}[/yellow]")

    finally:
        engine.close()


# ── stats 커맨드 ──────────────────────────────────────────


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """인덱스 통계를 출력합니다."""
    config_path = ctx.obj["config_path"]

    engine = get_engine(config_path)
    engine.open()

    try:
        s = engine.get_stats()

        table = Table(title="📊 인덱스 통계", show_header=False, border_style="blue")
        table.add_column("항목", style="bold")
        table.add_column("값", justify="right")

        table.add_row("총 문서 수", f"{s['total_documents']}개")
        table.add_row("총 청크 수", f"{s['total_chunks']}개")
        table.add_row("총 벡터 수", f"{s['total_vectors']}개")

        console.print(table)

    finally:
        engine.close()


# ── serve 커맨드 ──────────────────────────────────────────


@main.command()
@click.option("-h", "--host", type=str, default=None, help="바인드 호스트")
@click.option("-p", "--port", type=int, default=None, help="바인드 포트")
@click.pass_context
def serve(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """웹 UI 서버를 실행합니다."""
    from airgap_kor_search.config import load_or_create_config

    config_path = ctx.obj["config_path"]
    config = load_or_create_config(config_path)

    host = host or config.server.host
    port = port or config.server.port

    console.print(
        Panel(
            f"[bold green]🌐 웹 서버 시작[/bold green]\n\n"
            f"  URL: [cyan]http://{host}:{port}[/cyan]\n"
            f"  종료: Ctrl+C",
            border_style="green",
        )
    )

    import uvicorn

    from airgap_kor_search.server import create_app

    app = create_app(config_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ── init 커맨드 ───────────────────────────────────────────


@main.command()
@click.option(
    "-d", "--data-dir",
    type=click.Path(),
    default="./airgap_data",
    help="데이터 디렉토리 경로",
)
@click.pass_context
def init(ctx: click.Context, data_dir: str) -> None:
    """설정 파일과 디렉토리를 초기화합니다."""
    from airgap_kor_search.config import AppConfig

    data_path = Path(data_dir)
    config_path = data_path / "config.json"

    if config_path.exists():
        if not click.confirm(f"설정 파일이 이미 존재합니다: {config_path}\n덮어쓰시겠습니까?"):
            console.print("[dim]취소되었습니다.[/dim]")
            return

    config = AppConfig(data_dir=data_path)
    config.ensure_dirs()
    config.save(config_path)

    console.print("[green]✅ 초기화 완료[/green]")
    console.print()
    console.print(f"  설정 파일: [cyan]{config_path}[/cyan]")
    console.print(f"  데이터 디렉토리: [cyan]{data_path}[/cyan]")
    console.print(f"  모델 디렉토리: [cyan]{config.model.model_dir}[/cyan]")
    console.print()
    console.print(
        "💡 다음 단계: 모델 파일을 준비하세요.\n"
        "   [dim]자세한 내용: docs/model-preparation.md[/dim]"
    )


if __name__ == "__main__":
    main()
