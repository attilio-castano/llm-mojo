"""Validate launch inputs, prepare once, and hand inference to native Mojo."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from llm_mojo.models.qwen2.assets import verify_prepared
from llm_mojo.models.qwen2.tokenizer_assets import ensure_prepared
from llm_mojo.runtime.build import ensure_binary


def report_paths(report):
    if report is None:
        return None
    path = Path(report)
    sidecar = path.with_name(path.name + '.config.json')
    for item in (path, sidecar):
        if item.exists() or not item.parent.is_dir():
            raise ValueError('report must be a new file in an existing directory: ' + str(item))
    return sidecar


def prepare_launch(config, command):
    w = config.workload
    sidecar = report_paths(w.report)
    if w.system_file is not None:
        Path(w.system_file).read_text(encoding='utf-8')
    prepared, _ = verify_prepared(Path(w.prepared))
    tables = ensure_prepared(download=False)
    binary = ensure_binary(command, f'src/llm_mojo/cli/{command}_cli.mojo')
    if sidecar is not None:
        from llm_mojo.configuration import resolved_dict
        with sidecar.open('x') as output:
            json.dump(dict(command=command, configuration=resolved_dict(config)), output, indent=2)
            output.write('\n')
    return binary, str(prepared), str(tables)


def launch_chat(config):
    binary, prepared, tables = prepare_launch(config, 'chat')
    w = config.workload
    # Preserve the foreground process group; Mojo consumes SIGINT synchronously.
    os.execv(binary, [str(binary), prepared, tables, str(w.max_new_tokens), str(w.chunk_rows),
                     w.system_file or '', w.report or ''])


def launch_generate(config):
    w = config.workload
    if (w.prompt is None) == (w.prompt_file is None):
        raise ValueError('provide exactly one prompt or prompt file')
    text = w.prompt if w.prompt is not None else Path(w.prompt_file).read_text(encoding='utf-8')
    if not text:
        raise ValueError('prompt must not be empty')
    binary, prepared, tables = prepare_launch(config, 'generate')
    # Snapshot either input source; Python does no tokenization or inference.
    with tempfile.TemporaryDirectory(prefix='llm-mojo-prompt-') as directory:
        prompt = Path(directory) / 'prompt.txt'
        prompt.write_text(text, encoding='utf-8')
        subprocess.run([str(binary), prepared, tables, str(prompt), str(w.max_new_tokens),
                        str(w.chunk_rows), config.mode.name, *([w.report] if w.report else [])], check=True)
