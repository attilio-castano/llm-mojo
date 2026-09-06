import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from llm_mojo.benchmarks import profile as builder
from llm_mojo.benchmarks.attention_decode_contract import configuration, ENTRYPOINTS, VARIANTS
from llm_mojo.benchmarks.capture_trace import profile_contract, parse_target_identity, validate_target_identity
from llm_mojo.benchmarks.analyze_trace import segment_compute_commands
from llm_mojo.benchmarks import attention_prefill_contract as prefill
from llm_mojo.benchmarks import attention_sublayer_contract as sublayer

class AttentionProfileTests(unittest.TestCase):
    def test_sublayer_profile_binds_full_stage_sequence_and_causal_shape(self):
        p = self.profile()
        p.update(operation=sublayer.OPERATION, implementation='attention_sublayer_3',
                 entrypoint='enqueue_attention_sublayer', profile_iterations=300,
                 **sublayer.specification(3,64,4096))
        cfg, _, hardware = profile_contract(p)
        output = '\n'.join([
            'profile implementation: enqueue_attention_sublayer', 'device: Apple Test GPU',
            'api: metal', 'rows: 64', 'hidden: 896', 'warmup iterations: 100',
            'profile iterations: 300', 'post-profile idle milliseconds: 250',
            'profile workload: sublayer-r64-t4096-v3', 'profile dispatches per iteration: 12',
            'key value rows: 4096', 'query heads: 14', 'key value heads: 2'])
        target = parse_target_identity(output)
        validate_target_identity(target,cfg,hardware)
        for key, value in [('hidden_size',64), ('key_value_rows',65), ('dispatches_per_iteration',3)]:
            with self.assertRaises(ValueError):
                validate_target_identity({**target,key:value},cfg,hardware)
        for key, value in [('profile_rows',4097), ('profile_iterations',417),
                           ('implementation','attention_sublayer_0'), ('query_heads',True)]:
            with self.assertRaises(ValueError):
                sublayer.configuration({**p,key:value})
        self.assertEqual(len(sublayer.profile_grid(dict(variants=[3],workloads=sublayer.PROFILE_WORKLOADS))[1]),4)
        candidate = {**p, 'implementation':'attention_sublayer_4',
                     **sublayer.specification(4,64,4096)}
        sublayer.configuration(candidate)
        # The same source entrypoint serves both Wo mappings; the workload ID
        # must still prevent a rowwise capture being labeled as the candidate.
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            sublayer.configuration({**candidate,'profile_workload':p['profile_workload']})
        self.assertEqual(len(sublayer.profile_grid(dict(variants=[3,4],workloads=sublayer.PROFILE_WORKLOADS))[1]),8)
        with self.assertRaises(ValueError):
            sublayer.profile_grid(dict(variants=[3],workloads=sublayer.PROFILE_WORKLOADS[:-1]))

    def test_fp32_decode_profiles_bind_variant_specific_dispatches(self):
        for variant,dispatches in ((5,10),(6,11)):
            with self.subTest(variant=variant):
                p = self.profile()
                p.update(operation=sublayer.OPERATION,
                         implementation=f'attention_sublayer_{variant}',
                         entrypoint='enqueue_attention_sublayer', profile_iterations=450,
                         **sublayer.specification(variant,1,4096))
                cfg,_,_ = profile_contract(p)
                self.assertEqual(cfg['dispatches_per_iteration'],dispatches)
                for key,value in (('dispatches_per_iteration',12),
                                  ('profile_workload','sublayer-r1-t4096-v3'),
                                  ('profile_rows',2),('profile_iterations',501)):
                    with self.assertRaises(ValueError):
                        sublayer.configuration({**p,key:value})
        stages,grid = sublayer.profile_grid(dict(variants=[3,5,6],
                                               workloads=sublayer.DECODE_PROFILE_WORKLOADS))
        self.assertEqual([len(stages[v]) for v in (3,5,6)],[12,10,11])
        self.assertEqual(len(grid),6)
        with self.assertRaises(ValueError):
            sublayer.profile_grid(dict(variants=[3,5,6],workloads=sublayer.PROFILE_WORKLOADS))

    def test_fp32_sublayer_prefill_profile_binds_both_arms(self):
        p = self.profile()
        p.update(operation=sublayer.OPERATION, implementation='attention_sublayer_7',
                 entrypoint='enqueue_attention_sublayer', profile_iterations=450,
                 **sublayer.specification(7,64,4096))
        cfg,_,_ = profile_contract(p)
        self.assertEqual(cfg['dispatches_per_iteration'],10)
        for key,value in (('dispatches_per_iteration',12), ('profile_rows',1),
                          ('profile_workload','sublayer-r64-t4096-v4'),
                          ('profile_iterations',501)):
            with self.assertRaises(ValueError):
                sublayer.configuration({**p,key:value})
        stages,grid = sublayer.profile_grid(dict(variants=[4,7],
                                               workloads=sublayer.PREFILL_PROFILE_WORKLOADS))
        self.assertEqual([len(stages[v]) for v in (4,7)],[12,10])
        self.assertEqual(len(grid),6)
        for variants,workloads in (([3,7],sublayer.PREFILL_PROFILE_WORKLOADS),
                                  ([4,7],sublayer.PROFILE_WORKLOADS)):
            with self.assertRaises(ValueError):
                sublayer.profile_grid(dict(variants=variants,workloads=workloads))

    def test_integrated_attention_profile_binds_entrypoint_and_layout_copy(self):
        for r,t in sublayer.PROFILE_WORKLOADS:
            p = self.profile()
            p.update(operation=sublayer.OPERATION, implementation='attention_sublayer_9',
                     entrypoint='enqueue_attention_sublayer_integrated', profile_iterations=50,
                     **sublayer.specification(9,r,t))
            cfg, _, _ = profile_contract(p)
            self.assertEqual(cfg['dispatches_per_iteration'],9)
            for key,value in (('entrypoint','enqueue_attention_sublayer'),
                              ('dispatches_per_iteration',8),
                              ('implementation','attention_sublayer_8')):
                with self.assertRaises(ValueError):
                    sublayer.configuration({**p,key:value})
        stages,grid = sublayer.profile_grid(dict(variants=[8,9],workloads=sublayer.PROFILE_WORKLOADS))
        self.assertEqual(len(grid),8)
        self.assertEqual(stages[9][1:3],['packed QKV projection','QKV unpack'])
        self.assertEqual([len(stages[v]) for v in (8,9)],[10,9])

    def test_prefill_profile_binds_rectangular_shape_and_tile_ownership(self):
        for variant in prefill.VARIANTS:
            p = self.profile()
            p.update(operation=prefill.OPERATION, implementation=f'gqa_prefill_{variant}',
                     entrypoint=prefill.ENTRYPOINTS[f'gqa_prefill_{variant}'],
                     **prefill.specification(variant,64,4096))
            cfg, _, hardware = profile_contract(p)
            text = '\n'.join([
                f'profile implementation: {p["entrypoint"]}', 'device: Apple Test GPU',
                'api: metal', 'rows: 64', 'hidden: 64', 'warmup iterations: 100',
                'profile iterations: 500', 'post-profile idle milliseconds: 250',
                *[f'{label}: {p[k]}' for label,k in (
                    ('profile workload','profile_workload'),('profile dispatches per iteration','dispatches_per_iteration'),
                    ('key value rows','key_value_rows'),('query heads','query_heads'),
                    ('key value heads','key_value_heads'),('query tile','query_tile'),
                    ('key tile','key_tile'),('heads','heads'))]])
            target = parse_target_identity(text)
            validate_target_identity(target,cfg,hardware)
            for field in ('key_value_rows','query_tile','heads'):
                wrong = dict(target)
                wrong[field] += 1
                with self.assertRaises(ValueError):
                    validate_target_identity(wrong,cfg,hardware)
            for field,value in [('profile_rows',1),('key_value_rows',63),('query_tile',True),
                                ('profile_iterations',5001),('profile_warmup_iterations',101)]:
                with self.assertRaises(ValueError):
                    prefill.configuration({**p,field:value})

    def test_profile_build_requires_unchanged_source_and_repository(self):
        for change in (None, "source", "commit", "dirty"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                source = directory / "kernel.mojo"
                source.write_text("original source")
                binary = directory / "profile"
                provenance = directory / "profile.provenance.json"
                repo = {"commit": "a" * 40, "branch": "codex/test", "dirty": False}
                expected_repo = repo.copy()
                expected_sources = {source.name: builder.sha(source)}
                args = SimpleNamespace(
                    build_profile_binary=binary, profile_variant=9,
                    profile_rows=4096, profile_iterations=500,
                )

                def compile_binary(*args, **kwargs):
                    binary.write_bytes(b"compiled from original source")
                    if change == "source":
                        source.write_text("edited during compilation")
                    elif change == "commit":
                        repo["commit"] = "b" * 40
                    elif change == "dirty":
                        repo["dirty"] = True

                with patch.object(builder, "repository_state", side_effect=lambda: repo.copy()), \
                     patch.object(builder, "source_hashes", side_effect=lambda: {source.name: builder.sha(source)}), \
                     patch.object(builder, "stable_environment", return_value={
                         "hardware": {"chip": "Apple Test GPU", "gpu_api": "metal"}
                     }), \
                     patch.object(builder.subprocess, "run", side_effect=compile_binary):
                    if change is None:
                        builder.build_profile(args)
                        record = json.loads(provenance.read_text())
                        self.assertEqual(record["repository"], expected_repo)
                        self.assertEqual(record["source_sha256"], expected_sources)
                        self.assertEqual(record["binary"]["sha256"], builder.sha(binary))
                        profile_contract(record)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "source changed during profile build"):
                            builder.build_profile(args)
                        self.assertFalse(provenance.exists())

    def profile(self, variant=7):
        g, h, s = VARIANTS[variant]
        return {
            "schema_version": 1,
            "operation": "grouped_query_attention_decode",
            "implementation": f'gqa_decode_{variant}',
            "entrypoint": ENTRYPOINTS[f'gqa_decode_{variant}'],
            "profile_rows": 1,
            "hidden_size": 64,
            "key_value_rows": 256,
            "query_heads": 14,
            "key_value_heads": 2,
            "groups": g,
            "heads": h,
            "splits": s,
            "profile_workload": f'decode-t256-v{variant}',
            "dispatches_per_iteration": 3 if variant
            == 0 else (2 if s > 1 else 1),
            "profile_warmup_iterations": 100,
            "profile_iterations": 500,
            "profile_post_idle_milliseconds": 250,
            "repository": {
                "commit": "a" * 40,
                "dirty": False,
                "branch": "codex/test",
            },
            "hardware": {"chip": "Apple Test GPU", "gpu_api": "metal"},
        }

    def test_profile_runtime_matches_shape_parameters_and_dispatches(self):
        for variant in (0, 1, 7, 8, 9, 10, 11, 12):
            p = self.profile(variant)
            cfg, _, hardware = profile_contract(p)
            text = "\n".join(
                [
                    f'profile implementation: {p["entrypoint"]}',
                    "device: Apple Test GPU",
                    "api: metal",
                    "rows: 1",
                    "hidden: 64",
                    "warmup iterations: 100",
                    "profile iterations: 500",
                    "post-profile idle milliseconds: 250",
                ]
                + [
                    f'{label}: {p[k]}'
                    for label, k in (
                        ("profile workload", "profile_workload"),
                        (
                            "profile dispatches per iteration",
                            "dispatches_per_iteration",
                        ),
                        ("key value rows", "key_value_rows"),
                        ("query heads", "query_heads"),
                        ("key value heads", "key_value_heads"),
                        ("groups", "groups"),
                        ("heads", "heads"),
                        ("splits", "splits"),
                    )
                ]
            )
            target = parse_target_identity(text)
            validate_target_identity(target, cfg, hardware)
            target["key_value_rows"] = 16
            with self.assertRaises(ValueError):
                validate_target_identity(target, cfg, hardware)

    def test_profile_rejects_forged_shapes_counts_and_unbounded_runs(self):
        p = self.profile()
        for key, value in (
            ("splits", 4),
            ("key_value_rows", 0),
            ("query_heads", 2),
            ("profile_iterations", 3000),
            ("dispatches_per_iteration", 1),
            ("groups", True),
        ):
            bad = copy.deepcopy(p)
            bad[key] = value
            with self.assertRaises(ValueError):
                configuration(bad)
        setup, correctness, warmup, profile = segment_compute_commands(
            list(range(17)), 2, 5, 2, False
        )
        self.assertEqual(
            (len(setup), len(correctness), len(warmup), len(profile)),
            (3, 0, 4, 10),
        )
