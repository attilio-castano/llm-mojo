"""Build a bounded GQA profile binary for receipt-verified Metal capture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .._repository import repository_root
from .attention_decode_contract import VARIANTS
from . import attention_prefill_contract as prefill
from . import attention_sublayer_contract as sublayer
from .environment import ensure_record_location, repository_state, stable_environment, utc_now
from .study import sha
from .run import source_hashes

def name(variant):
    return "materialized" if variant == 0 else (
        ("g%d-h%d-s%d" % VARIANTS[variant])
        + ("-conditional" if variant in (11, 12) else "")
    )


def specification(variant, rows):
    g, h, s = VARIANTS[variant]
    return {
        "id": variant,
        "name": name(variant),
        "groups": g,
        "conditional_rescale": variant in (11, 12),
        "heads": h,
        "splits": s,
        "dispatches": 3 if variant == 0 else (2 if s > 1 else 1),
        "workspace_bytes": rows * 14 * 2 if variant
        == 0 else (14 * s * 66 * 4 if s > 1 else 0),
    }


def build_profile(args):
    binary = args.build_profile_binary.resolve()
    ensure_record_location(binary)
    if binary.exists() or Path(str(binary) + ".provenance.json").exists():
        raise RuntimeError("refusing to overwrite profile artifacts")
    repo = repository_state()
    operation = getattr(args,'operation','gqa_decode')
    query_rows = getattr(args,'profile_query_rows',1)
    is_prefill = operation=='gqa_prefill'
    is_sublayer = operation == sublayer.OPERATION
    rectangular = is_prefill or is_sublayer
    warmup = getattr(args,'profile_warmup',100)
    allowed = sublayer.VARIANTS if is_sublayer else (prefill.VARIANTS if is_prefill else VARIANTS)
    if args.profile_variant not in allowed:
        raise RuntimeError('unknown profile variant for operation')
    spec = (dict(dispatches=len(sublayer.STAGES)) if is_sublayer else
            dict(dispatches=3 if args.profile_variant<=1 else 1) if is_prefill
            else specification(args.profile_variant, args.profile_rows))
    if (
        repo["dirty"]
        or not 0 <= warmup <= 100
        or not 1 <= query_rows <= args.profile_rows
        or (not rectangular and query_rows != 1)
        or (is_prefill and args.profile_variant not in prefill.VARIANTS)
        or not 1 <= args.profile_rows <= 4096
        or not 1 <= args.profile_iterations * spec["dispatches"] <= 5000
    ):
        raise RuntimeError(
            "profile requires clean source, bounded shape and dispatch count"
        )
    sources = source_hashes()
    fixtures = sublayer.fixture_identity() if is_sublayer else None
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "build",
        "-I",
        "src",
        "-D",
        f'GQA_PROFILE_ROWS={args.profile_rows}',
        "-D",
        f'GQA_PROFILE_VARIANT={args.profile_variant}',
        "-D",
        f'GQA_PROFILE_ITERATIONS={args.profile_iterations}',
        "-D",
        f'GQA_PROFILE_QUERY_ROWS={query_rows}',
        "-D",
        f'GQA_PROFILE_WARMUP={warmup}',
        "src/llm_mojo/benchmarks/attention_sublayer.mojo" if is_sublayer else (
            "src/llm_mojo/benchmarks/attention_prefill.mojo" if is_prefill else "src/llm_mojo/benchmarks/attention_decode.mojo"),
        "-o",
        str(binary),
    ]
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    subprocess.run(command, cwd=repository_root(), check=True, env=environment)
    if repository_state() != repo or source_hashes() != sources:
        raise RuntimeError("source changed during profile build")
    if is_sublayer and sublayer.fixture_identity() != fixtures:
        raise RuntimeError('attention fixtures changed during profile build')
    workload = (sublayer.specification(args.profile_variant,query_rows,args.profile_rows) if is_sublayer else
                prefill.specification(args.profile_variant,query_rows,args.profile_rows) if is_prefill else {
        "groups": spec["groups"], "heads": spec["heads"], "splits": spec["splits"],
        "conditional_rescale": spec["conditional_rescale"]})
    record = {
        "schema_version": 1,
        "operation": sublayer.OPERATION if is_sublayer else (prefill.OPERATION if is_prefill else "grouped_query_attention_decode"),
        "repository": repo,
        **stable_environment(),
        "created_utc": utc_now(),
        "implementation": f'{operation}_{args.profile_variant}',
        "entrypoint": 'enqueue_attention_sublayer' if is_sublayer else prefill.ENTRYPOINTS[f'gqa_prefill_{args.profile_variant}'] if is_prefill else (
            "enqueue_grouped_query_attention_apple_gpu" if args.profile_variant==0 else "enqueue_grouped_query_attention_decode_apple_gpu"),
        "profile_rows": 1,
        "hidden_size": 64,
        "key_value_rows": args.profile_rows,
        "query_heads": 14,
        "key_value_heads": 2,
        "profile_workload": f'decode-t{args.profile_rows}-v{args.profile_variant}',
        "dispatches_per_iteration": spec["dispatches"],
        "profile_warmup_iterations": warmup,
        "profile_iterations": args.profile_iterations,
        "profile_post_idle_milliseconds": 250,
        **workload,
        "source_sha256": sources,
        "binary": {"bytes": binary.stat().st_size, "sha256": sha(binary)},
        "command": command[:-1] + ["<external-profile-binary>"],
    }
    if is_sublayer:
        record['attention_fixtures'] = fixtures
    Path(str(binary) + ".provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=['gqa_decode','gqa_prefill','attention_sublayer'], default='gqa_decode')
    p.add_argument('--profile-query-rows', type=int, default=1)
    p.add_argument('--profile-warmup', type=int, default=100)
    p.add_argument('--build-profile-binary', type=Path, required=True)
    p.add_argument('--profile-variant', type=int, choices=sorted(set(VARIANTS)|set(prefill.VARIANTS)), default=9)
    p.add_argument('--profile-rows', type=int, default=4096)
    p.add_argument('--profile-iterations', type=int, default=500)
    build_profile(p.parse_args())
