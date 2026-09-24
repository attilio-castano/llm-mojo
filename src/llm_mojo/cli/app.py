"""Public commands; configuration and launch services own execution details."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Annotated

import typer

from llm_mojo.configuration import MODEL, resolve_run, resolve_bench, resolved_dict

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False,
                  help='Run and study native Mojo inference on Apple Silicon.')
models = typer.Typer(no_args_is_help=True, help='Supported models and explicit asset preparation.')
bench = typer.Typer(no_args_is_help=True, help='Existing paired studies and their evidence gates.')
app.add_typer(models, name='models')
app.add_typer(bench, name='bench')

Preset = Annotated[str, typer.Option(help='Workload preset: interactive, short, whole-prompt.')]
Model = Annotated[str | None, typer.Option(help='Supported model identifier; Qwen is the default.')]
Mode = Annotated[str | None, typer.Option(help='Application execution mode; supported: fast.')]
Inspect = Annotated[bool, typer.Option('--show-config', help='Print resolved configuration without loading or compiling.')]


def configuration(command, **kwargs):
    try:
        return resolve_run(command, **kwargs)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def show(config):
    typer.echo(json.dumps(resolved_dict(config), indent=2))


@app.command()
def setup(offline: Annotated[bool, typer.Option('--offline', help='Never download; use the store or verified local copies.')] = False,
          check: Annotated[bool, typer.Option('--check', help='Only report readiness; change nothing.')] = False,
          import_from: Annotated[list[Path] | None, typer.Option(
              '--import-from', help='A checkout or model directory holding verified assets.')] = None,
          store: Annotated[Path | None, typer.Option(
              help='Shared store (default: LLM_MOJO_CACHE_DIR, else ~/.cache/llm-mojo).')] = None,
          build: Annotated[bool, typer.Option('--build/--no-build', help='Build the chat and generate executables.')] = True):
    """Check the toolchain, prepare the model once per machine, and build chat."""
    from llm_mojo.models.qwen2.assets import setup as run_setup
    status = run_setup(store, offline=offline, check=check, import_from=import_from or (), build=build,
                       log=typer.echo)
    if status:
        raise typer.Exit(status)


@app.command()
def chat(preset: Preset = 'interactive', model: Model = None, mode: Mode = None,
         prepared: Path | None = None, max_new_tokens: int | None = None,
         chunk_rows: int | None = None, system_file: Path | None = None,
         report: Path | None = None, show_config: Inspect = False):
    """Start a native conversation with persistent KV caches."""
    cfg = configuration('chat', preset=preset, model=model, mode=mode,
                        prepared=prepared, max_new_tokens=max_new_tokens,
                        chunk_rows=chunk_rows, system_file=system_file, report=report)
    if show_config:
        show(cfg)
        return
    from llm_mojo.runtime.launch import launch_chat
    launch_chat(cfg)


@app.command()
def generate(preset: Preset = 'interactive', model: Model = None, mode: Mode = None,
             prompt: str | None = None, prompt_file: Path | None = None,
             prepared: Path | None = None, max_new_tokens: int | None = None,
             chunk_rows: int | None = None, report: Path | None = None,
             show_config: Inspect = False):
    """Generate from raw text (without a chat template)."""
    cfg = configuration('generate', preset=preset, model=model, mode=mode,
                        prompt=prompt, prompt_file=prompt_file, prepared=prepared,
                        max_new_tokens=max_new_tokens, chunk_rows=chunk_rows, report=report)
    if show_config:
        show(cfg)
        return
    if prompt is None and prompt_file is None:
        raise typer.BadParameter('provide --prompt or --prompt-file')
    from llm_mojo.runtime.launch import launch_generate
    launch_generate(cfg)


@models.command('list')
def list_models(prepared: Path | None = None):
    """List supported capabilities, shared-store state and verified local preparation status."""
    from llm_mojo.models.qwen2 import assets
    report = assets.status()
    path = prepared.resolve() if prepared else assets.prepared_directory()
    try:
        assets.verify_prepared(path)
        status = 'ready' if report['tokenizer_tables'] == 'ready' else 'tokenizer preparation required'
    except (OSError, ValueError, KeyError, TypeError) as error:
        status = 'preparation required: ' + str(error)
    typer.echo(json.dumps(dict(**assets.capabilities(), prepared=str(path), status=status, **report), indent=2))


@models.command('prepare')
def prepare_model(model: Annotated[str, typer.Argument()] = MODEL, output: Path | None = None,
                  download: Annotated[bool, typer.Option(help='Download pinned assets only when missing.')] = False):
    """Prepare Qwen in the shared store and link this checkout; opt in to missing downloads."""
    if model != MODEL:
        raise typer.BadParameter('unsupported model: ' + model)
    from llm_mojo.models.qwen2.assets import prepare
    prepare(output, download=download)


@bench.command('list')
def list_studies():
    """List named presets and maintained studies."""
    from llm_mojo.configuration import BENCH_PRESETS
    from llm_mojo.benchmarks.study import STUDIES
    typer.echo(json.dumps(dict(presets={k: v.studies for k, v in BENCH_PRESETS.items()},
                              studies=list(STUDIES)), indent=2))


@bench.command('build')
def build_bench(build_dir: Annotated[Path, typer.Option()]):
    """Build the existing study executables from a clean commit."""
    from llm_mojo.benchmarks.run import build
    build(build_dir.resolve())


@bench.command('run')
def run_bench(preset: str = 'core', study: Annotated[list[str] | None, typer.Option('--study')] = None,
              build_dir: Path | None = None, output: Path | None = None,
              parallelism_screen: Path | None = None, tile_screen: Path | None = None,
              tile_kernel_screen: Path | None = None, mlp_decode_screen: Path | None = None,
              decoder_screen: Path | None = None, policy_confirmation: Path | None = None,
              show_config: Inspect = False):
    """Run complete existing study grids with their original evidence checks."""
    try:
        cfg = resolve_bench(preset, studies=study, build_dir=build_dir, output=output,
                            parallelism_screen=parallelism_screen, tile_screen=tile_screen,
                            tile_kernel_screen=tile_kernel_screen, mlp_decode_screen=mlp_decode_screen,
                            decoder_screen=decoder_screen, policy_confirmation=policy_confirmation)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if show_config:
        show(cfg)
        return
    if cfg.build_dir is None or cfg.output is None:
        raise typer.BadParameter('run requires --build-dir and --output')
    from llm_mojo.benchmarks.run import run
    options = {k: Path(v) if v is not None else None for k, v in asdict(cfg).items()
               if k not in ('studies', 'build_dir', 'output')}
    run(Path(cfg.build_dir), Path(cfg.output), cfg.studies, resolved_configuration=resolved_dict(cfg), **options)


@bench.command('select-decoder')
def select_decoder(build_dir: Annotated[Path, typer.Option()],
                   decoder_screen: Annotated[Path, typer.Option()]):
    """Freeze selection from a complete decoder screen."""
    from llm_mojo.benchmarks.run import main
    main(['select-decoder', '--build-dir', str(build_dir.resolve()),
          '--decoder-screen', str(decoder_screen.resolve())])


@bench.command('confirm-decoder')
def confirm_decoder(build_dir: Annotated[Path, typer.Option()],
                    decoder_screen: Annotated[Path, typer.Option()],
                    output: Annotated[Path, typer.Option()]):
    """Confirm decoder selection against its complete qualification run."""
    from llm_mojo.benchmarks.run import main
    main(['confirm-decoder', '--build-dir', str(build_dir.resolve()),
          '--decoder-screen', str(decoder_screen.resolve()), '--output', str(output.resolve())])


@bench.command('tokenizer')
def tokenizer_bench(operation: Annotated[str, typer.Argument()],
                    build_dir: Annotated[Path, typer.Option()], output: Path | None = None):
    """Build, run or report the existing tokenizer study."""
    if operation not in ('build', 'run', 'report'):
        raise typer.BadParameter('operation must be build, run or report')
    from llm_mojo.benchmarks.run import main
    main([operation + '-tokenizer', '--build-dir', str(build_dir.resolve()),
          *(['--output', str(output.resolve())] if output is not None else [])])


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True}, add_help_option=False)
def tokenizer(ctx: typer.Context):
    """Tokenizer setup/encode/decode; --help lists the existing options."""
    from llm_mojo.models.qwen2.tokenizer_assets import main
    main(ctx.args)


@app.command()
def validate(prepare_only: bool = False):
    """Run the repository's frozen-oracle, Python, Mojo and smoke validation."""
    from llm_mojo.validation.suite import main
    main(['--prepare-only'] if prepare_only else [])


def main():
    try:
        app()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        typer.echo('Error: ' + str(error), err=True)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
