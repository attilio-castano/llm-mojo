"""Typed Hydra composition for application workloads and existing study selections."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from llm_mojo.models.qwen2.assets import (prepared_directory, MODEL_ID as MODEL,
                                         REVISION, CHECKPOINT_SHA, CONTEXT_CAPACITY, APPLICATION_MODE,
                                         GENERATION_MODES)


@dataclass
class ModelConfig:
    name: str = MODEL
    revision: str = REVISION
    checkpoint_sha256: str = CHECKPOINT_SHA


@dataclass
class ModeConfig:
    name: str = APPLICATION_MODE


@dataclass
class WorkloadConfig:
    max_new_tokens: int = 256
    chunk_rows: int = 256
    prepared: str | None = None
    prompt: str | None = None
    prompt_file: str | None = None
    system_file: str | None = None
    report: str | None = None


@dataclass
class RunConfig:
    defaults: list = field(default_factory=lambda: [
        {'model': MODEL}, {'mode': 'fast'}, {'workload': 'interactive'}, '_self_'])
    model: ModelConfig = MISSING
    mode: ModeConfig = MISSING
    workload: WorkloadConfig = MISSING


@dataclass
class BenchConfig:
    studies: list[str] = field(default_factory=lambda: ['rms_norm', 'linear_decode'])
    build_dir: str | None = None
    output: str | None = None
    parallelism_screen: str | None = None
    tile_screen: str | None = None
    tile_kernel_screen: str | None = None
    mlp_decode_screen: str | None = None
    policy_confirmation: str | None = None


WORKLOADS = {'interactive': WorkloadConfig(),
             'short': WorkloadConfig(max_new_tokens=32, chunk_rows=64),
             'whole-prompt': WorkloadConfig(chunk_rows=0)}
BENCH_PRESETS = {'core': BenchConfig(),
                 'attention': BenchConfig(studies=['gqa_decode', 'gqa_prefill'])}


def _register():
    store = ConfigStore.instance()
    store.store(name='llm_run', node=RunConfig)
    store.store(group='model', name=MODEL, node=ModelConfig)
    for name in GENERATION_MODES:
        store.store(group='mode', name=name, node=ModeConfig(name=name))
    for name, node in WORKLOADS.items():
        store.store(group='workload', name=name, node=node)
    for name, node in BENCH_PRESETS.items():
        store.store(name='llm_bench_' + name, node=node)


_register()


def resolve_run(command, *, preset='interactive', model=None, mode=None, **options):
    if command not in ('chat', 'generate'):
        raise ValueError('unknown application command')
    if preset not in WORKLOADS:
        raise ValueError('unknown workload preset: ' + preset)
    if model is not None and model != MODEL:
        raise ValueError('unsupported model: ' + model)
    modes = GENERATION_MODES if command == 'generate' else (APPLICATION_MODE,)
    if mode is not None and mode not in modes:
        raise ValueError(f'unsupported {command} mode: {mode}; supported: ' + ', '.join(modes))
    with initialize(version_base='1.3', config_path=None):
        cfg = compose(config_name='llm_run', overrides=['workload=' + preset, 'mode=' + (mode or APPLICATION_MODE)])
    result = OmegaConf.to_object(cfg)
    w = result.workload
    # Literal CLI text never enters OmegaConf interpolation or override grammar.
    for key, value in options.items():
        if key not in WorkloadConfig.__dataclass_fields__:
            raise ValueError('unknown workload option: ' + key)
        if value is not None:
            if key in ('max_new_tokens', 'chunk_rows'):
                if type(value) is not int:
                    raise ValueError(key + ' must be an integer')
            elif not isinstance(value, (str, Path)):
                raise ValueError(key + ' must be text or a path')
            setattr(w, key, str(value) if isinstance(value, Path) else value)
    minimum = 1 if command == 'chat' else 0
    if not minimum <= w.max_new_tokens <= CONTEXT_CAPACITY or not minimum <= w.chunk_rows <= CONTEXT_CAPACITY:
        raise ValueError(f'{command} reply and chunk limits must be in {minimum}..{CONTEXT_CAPACITY}')
    if command == 'chat' and (w.prompt is not None or w.prompt_file is not None):
        raise ValueError('chat reads messages interactively')
    if command == 'generate' and w.system_file is not None:
        raise ValueError('plain-text generation does not apply a system message')
    if w.prompt is not None and w.prompt_file is not None:
        raise ValueError('use exactly one of --prompt and --prompt-file')
    w.prepared = str(Path(w.prepared).resolve() if w.prepared else prepared_directory())
    for key in ('prompt_file', 'system_file', 'report'):
        if getattr(w, key) is not None:
            setattr(w, key, str(Path(getattr(w, key)).resolve()))
    return result


def resolve_bench(preset='core', **options):
    if preset not in BENCH_PRESETS:
        raise ValueError('unknown benchmark preset: ' + preset)
    with initialize(version_base='1.3', config_path=None):
        cfg = compose(config_name='llm_bench_' + preset)
    result = OmegaConf.to_object(cfg)
    for key, value in options.items():
        if key not in BenchConfig.__dataclass_fields__:
            raise ValueError('unknown benchmark option: ' + key)
        if value is not None:
            if key == 'studies':
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ValueError('studies must be a list of names')
                setattr(result, key, value.copy())
            else:
                setattr(result, key, str(Path(value).resolve()))
    from llm_mojo.benchmarks.study import STUDIES, REPLAY_ONLY
    from llm_mojo.benchmarks.decoder_layer_contract import MEASUREMENT_VARIANTS, RUNNABLE_VARIANTS
    names = (set(STUDIES) - REPLAY_ONLY) | {f'decoder_policies_cost_{v}_{m}' for v in MEASUREMENT_VARIANTS & RUNNABLE_VARIANTS
                                            for m in ('hot', 'ring')}
    if not result.studies or len(set(result.studies)) != len(result.studies):
        raise ValueError('select at least one study, with no duplicates')
    if set(result.studies) - names:
        raise ValueError('unknown studies: ' + ', '.join(sorted(set(result.studies) - names)))
    return result


def resolved_dict(config):
    result = asdict(config)
    result.pop('defaults', None)
    return result
