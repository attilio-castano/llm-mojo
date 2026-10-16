"""The presentation must remain reproducible from its recorded evidence.

These checks use the retained raw records as independent oracles. They do not
run a model, compile Mojo, or turn a browser illustration into compiler evidence.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import gzip
import hashlib
from html.parser import HTMLParser
from importlib import util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPLORER = Path("docs/explorer")
SPEC = util.spec_from_file_location("explorer_build", ROOT / EXPLORER / "build.py")
builder = util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def paired_samples(archive, history, comparison):
    """Derive the report statistic without using the explorer's aggregation."""
    groups = defaultdict(list)
    for sample in archive["timing"]["samples"]:
        if (sample["prefix"], sample["comparison"]) == (history, comparison):
            groups[sample["block"], sample["arm"]].append(sample)
    arms = sorted({arm for _, arm in groups})
    if len(arms) != 2 or len(groups) != 8:
        raise AssertionError("Expected four complete pairs")
    medians = []
    for arm in arms:
        values = []
        for block in range(4):
            samples = groups[block, arm]
            if sorted(s["sample"] for s in samples) != list(range(10)):
                raise AssertionError("Expected ten retained samples in each arm")
            values.append(statistics.median(s["elapsed_ns"] for s in samples) / 1e6)
        medians.append(values)
    ratios = [candidate / control for control, candidate in zip(*medians)]
    return medians, ratios


class RuntimeResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag in {"script", "img", "iframe", "audio", "video", "source"} and "src" in attributes:
            self.urls.append(attributes["src"])
        if tag == "link" and attributes.get("rel") in {"stylesheet", "preload", "modulepreload"}:
            self.urls.append(attributes.get("href", ""))


class ExplorerEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_json(ROOT / EXPLORER / "recordings.json")
        cls.data = builder.build_data(ROOT)
        cls.archives = {
            name: read_json(ROOT / cls.manifest["assets"][name]["path"])
            for name in ("residualNorm", "projectionArrangements", "engineOffline")
        }

    @contextmanager
    def copied_inputs(self):
        """Copy only the standalone builder and its declared inputs."""
        with tempfile.TemporaryDirectory(prefix="llm-mojo-explorer-") as directory:
            root = Path(directory)
            shutil.copytree(ROOT / EXPLORER, root / EXPLORER,
                            ignore=shutil.ignore_patterns("__pycache__"))
            paths = set(self.manifest["sources"])
            for asset in self.manifest["assets"].values():
                paths.add(asset["path"])
                if "hashManifest" in asset:
                    paths.add(asset["hashManifest"])
            receipt = read_json(ROOT / EXPLORER / "contract-receipt.json")
            paths.update(receipt["sources"])
            for relative in paths:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            yield root

    def check_cli(self, root):
        return subprocess.run(
            [sys.executable, "-B", str(root / EXPLORER / "build.py"), "--check"],
            cwd=root, capture_output=True, text=True, timeout=45,
        )

    def test_generated_page_is_current_and_deterministic(self):
        first = builder.render(ROOT)
        self.assertEqual(first, builder.render(ROOT))
        self.assertEqual((ROOT / EXPLORER / "index.html").read_text(), first)
        checked = self.check_cli(ROOT)
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_offline_page_embeds_its_runtime_resources(self):
        html = (ROOT / EXPLORER / "index.html").read_text()
        resources = RuntimeResources()
        resources.feed(html)
        self.assertEqual([url for url in resources.urls if not url.startswith("data:")], [])
        self.assertNotIn("__EXPLORER_DATA__", html)
        self.assertNotRegex(html, r"@import\s|url\(\s*['\"]?https?://")

    def test_check_rejects_stale_html_without_overwriting_it(self):
        with self.copied_inputs() as root:
            page = root / EXPLORER / "index.html"
            changed = page.read_text() + "\n<!-- stale -->\n"
            page.write_text(changed)
            checked = self.check_cli(root)
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn("stale", checked.stderr)
            self.assertEqual(page.read_text(), changed)

    def test_modified_pinned_source_and_asset_are_rejected(self):
        source = next(path for path in self.manifest["sources"] if path.endswith(".mojo"))
        asset = self.manifest["assets"]["residualNorm"]["path"]
        for relative in (source, asset, self.manifest["vendor"]["sourcePath"]):
            with self.subTest(path=relative), self.copied_inputs() as root:
                path = root / relative
                path.write_bytes(path.read_bytes() + b"\nchanged")
                checked = self.check_cli(root)
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn("checksum mismatch", checked.stderr)

    def test_rehashed_summary_cannot_replace_the_raw_measurement(self):
        with self.copied_inputs() as root:
            manifest = copy.deepcopy(self.manifest)
            asset = manifest["assets"]["residualNormSummary"]
            path = root / asset["path"]
            summary = read_json(path)
            summary["contexts"][0]["comparisons"][0]["median_reduction"] += 0.1
            path.write_text(json.dumps(summary))
            asset["sha256"] = sha(path.read_bytes())
            (root / EXPLORER / "recordings.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "summary disagrees"):
                builder.build_data(root)

    def test_invalid_source_line_range_is_rejected(self):
        with self.copied_inputs() as root:
            manifest = copy.deepcopy(self.manifest)
            source = next(iter(manifest["sources"].values()))
            source["startLine"] = 1000000
            source["endLine"] = 1000001
            (root / EXPLORER / "recordings.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source excerpt range"):
                builder.build_data(root)

    def test_compiler_receipt_does_not_accept_changed_sources_or_fake_success(self):
        with self.copied_inputs() as root:
            path = root / "uv.lock"
            path.write_bytes(path.read_bytes() + b"\nchanged\n")
            with self.assertRaisesRegex(ValueError, "compiler receipt source checksum mismatch"):
                builder.build_data(root)
        with self.copied_inputs() as root:
            path = root / EXPLORER / "contract-receipt.json"
            receipt = read_json(path)
            receipt["executions"] = []
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "compiler receipt"):
                builder.build_data(root)

    def test_source_excerpts_and_links_use_valid_pinned_ranges(self):
        self.assertGreater(len(self.data["sources"]), 10)
        for path, record in self.data["sources"].items():
            with self.subTest(path=path):
                source = (ROOT / path).read_bytes()
                lines = source.decode().splitlines()
                self.assertEqual(sha(source), record["sha256"])
                self.assertRegex(record["revision"], r"^[0-9a-f]{40}$")
                self.assertGreaterEqual(record["startLine"], 1)
                self.assertLessEqual(record["startLine"], record["endLine"])
                self.assertLessEqual(record["endLine"], len(lines))
                self.assertEqual(record["excerpt"], "\n".join(
                    lines[record["startLine"] - 1:record["endLine"]]))
                self.assertEqual(record["url"],
                                 f'{self.manifest["repository"]}/blob/{record["revision"]}/{path}'
                                 f'#L{record["startLine"]}')

    def test_chat_uses_one_recorded_turn_and_preserves_pending_token(self):
        archive = self.archives["residualNorm"]
        arm = next(arm for arm in archive["terminal"]["blocks"][0]["arms"]
                   if arm["selection"] == 3)
        turn = next(turn for turn in arm["turns"] if turn["turn"] == 1)
        chat = self.data["chat"]
        self.assertEqual(chat["promptIds"], turn["prompt"])
        self.assertEqual(chat["generatedIds"], turn["generated"])
        self.assertEqual(chat["historyIds"], turn["history"])
        self.assertEqual(len(chat["promptIds"]), 44)
        self.assertEqual(len(chat["generatedIds"]), 128)
        self.assertEqual(chat["closingIds"], [151645, 198])
        self.assertEqual(chat["historyIds"],
                         chat["promptIds"] + chat["generatedIds"] + chat["closingIds"])
        self.assertEqual(chat["cached"] + 1,
                         len(chat["promptIds"]) + len(chat["generatedIds"]))
        self.assertEqual(chat["cachedBefore"], 0)
        generated = b"".join(bytes(token["bytes"]) for token in chat["tokens"])
        self.assertEqual(chat["text"], generated.decode("utf-8"))
        self.assertEqual([token["id"] for token in chat["tokens"]], turn["generated"])
        token_events = [event for event in chat["events"] if event["event"] == "token"]
        text_events = [event for event in chat["events"] if event["event"] == "text"]
        self.assertEqual([int(event["value"]) for event in token_events], turn["generated"])
        self.assertEqual([int(event["value"]) for event in text_events],
                         [len(token["bytes"]) for token in chat["tokens"]])
        self.assertAlmostEqual(chat["firstTokenMs"], token_events[0]["nanoseconds"] / 1e6)
        self.assertAlmostEqual(chat["firstVisibleMs"], text_events[0]["nanoseconds"] / 1e6)
        self.assertEqual(chat["provenance"]["revision"],
                         archive["timing"]["build"]["source"]["repository"]["commit"])

    def test_serving_replay_is_complete_and_returns_all_blocks(self):
        archive = self.archives["engineOffline"]
        run = next(run for run in archive["runs"]
                   if run["arm"] == "chunked" and not run["calibration"])
        original = run["parsed"]
        serving = self.data["serving"]
        self.assertEqual(serving["steps"], original["steps"])
        self.assertEqual(serving["events"], original["events"])
        self.assertEqual(serving["requests"], archive["trace"]["requests"])
        self.assertEqual(serving["drained"], original["drained"])
        steps, events, requests = serving["steps"], serving["events"], serving["requests"]
        tokens = [event for event in events if event["kind"] == "token"]
        finishes = [event for event in events if event["kind"] == "finish"]
        self.assertEqual(len(steps), 46)
        self.assertEqual(len(requests), 8)
        self.assertEqual(len(tokens), 256)
        self.assertEqual(len(finishes), 8)
        self.assertEqual(Counter(event["request_id"] for event in tokens),
                         {request["request_id"]: request["max_new_tokens"] for request in requests})
        self.assertEqual(sum(step["prefill_tokens"] for step in steps),
                         sum(len(request["prompt_ids"]) for request in requests))
        self.assertEqual(sum(step["decode_seqs"] for step in steps), len(tokens) - len(requests))
        self.assertEqual(sum(step["finished"] for step in steps), len(requests))
        self.assertTrue(any(step["prefill_tokens"] and step["decode_seqs"] for step in steps))
        for step in steps:
            self.assertEqual(step["total_tokens"], step["prefill_tokens"] + step["decode_seqs"])
            self.assertLessEqual(step["total_tokens"], serving["tokenBudget"])
            self.assertTrue(0 <= step["blocks_free"] <= serving["blocks"])
        self.assertEqual(steps[-1]["waiting"], 0)
        self.assertEqual(steps[-1]["blocks_free"], serving["drained"]["free_blocks"])
        self.assertEqual(serving["drained"]["free_blocks"], serving["blocks"])

    def check_study_row(self, row, archive, comparison):
        medians, ratios = paired_samples(archive, row["history"], comparison)
        _, calibration = paired_samples(archive, row["history"], 0)
        reduction = 1 - statistics.median(ratios)
        threshold = max(0.05, *(abs(ratio - 1) for ratio in calibration))
        self.assertEqual(row["pairedRatios"], ratios)
        self.assertEqual(row["controlMs"], statistics.median(medians[0]))
        self.assertEqual(row["candidateMs"], statistics.median(medians[1]))
        self.assertAlmostEqual(row["reductionPct"], 100 * reduction)
        self.assertAlmostEqual(row["requiredPct"], 100 * threshold)
        self.assertEqual(row["qualifies"], all(r < 1 for r in ratios) and reduction > threshold)

    def test_study_decisions_follow_paired_raw_samples_at_every_context(self):
        studies = self.data["studies"]
        for row in studies["accepted"]["rows"]:
            self.check_study_row(row, self.archives["residualNorm"], 3)
            self.assertTrue(row["qualifies"])
        self.assertEqual([row["history"] for row in studies["accepted"]["rows"]], [64, 1024, 3968])
        self.assertEqual(len(studies["rejected"]["rows"]), 5)
        for candidate in studies["rejected"]["rows"]:
            self.assertFalse(candidate["qualifies"])
            for row in candidate["cells"]:
                self.check_study_row(row, self.archives["projectionArrangements"], candidate["id"])
        for name, arm in (("accepted", "residualNorm"), ("rejected", "projectionArrangements")):
            source = self.archives[arm]["timing"]["build"]["source"]["repository"]["commit"]
            self.assertEqual(studies[name]["provenance"]["revision"], source)
            self.assertEqual(len(self.archives[arm]["timing"]["samples"]), 1440)

    def test_plan_root_coverage_allows_repetition_but_rejects_missing_input(self):
        plans = {plan["id"]: plan for plan in self.data["proof"]["plans"]}

        def leaves(plan, index):
            node = plan["nodes"][index]
            if node["kind"] == "empty":
                return Counter()
            if node["kind"] == "input":
                return Counter([node["input"]])
            self.assertLess(node["left"], index)
            self.assertLess(node["right"], index)
            return leaves(plan, node["left"]) + leaves(plan, node["right"])

        for plan in plans.values():
            coverage = leaves(plan, plan["root"])
            self.assertEqual(plan["support"], sorted(coverage))
            self.assertEqual(plan["missing"], sorted(set(range(8)) - set(coverage)))
            self.assertEqual(plan["accepted"], set(coverage) == set(range(8)))
        for name in ("sequential_plan", "balanced_plan", "earlier_root_plan"):
            self.assertEqual(leaves(plans[name], plans[name]["root"]), Counter(range(8)))
        repeated = plans["repeated_permuted_plan"]
        self.assertTrue(repeated["accepted"])
        self.assertEqual(leaves(repeated, repeated["root"]), {i: 2 for i in range(8)})
        missing = plans["invalid_plan"]
        self.assertFalse(missing["accepted"])
        self.assertEqual(missing["missing"], [7])

    def test_displayed_candidate_records_match_the_independent_two_pass_oracle(self):
        proof = self.data["proof"]
        inputs = proof["inputs"]
        self.assertEqual(len(inputs), 8)
        top = max(record["rank"] for record in inputs)
        expected = {
            "rank": top,
            "token": min(record["token"] for record in inputs if record["rank"] == top),
            "invalid": int(any(record["invalid"] for record in inputs)),
        }
        self.assertEqual(expected, {"rank": 3, "token": 31, "invalid": 1})
        self.assertEqual(proof["expected"], expected)
        self.assertTrue(any(record["invalid"] and record["rank"] < top for record in inputs))
        self.assertGreater(sum(record["rank"] == top for record in inputs), 1)

    def test_contract_receipt_binds_real_compiler_outcomes_to_current_sources(self):
        receipt = read_json(ROOT / EXPLORER / "contract-receipt.json")
        self.assertEqual(receipt["status"], "passed")
        self.assertTrue(receipt["sources_unchanged_during_run"])
        for path, expected in receipt["sources"].items():
            with self.subTest(path=path):
                self.assertEqual(sha((ROOT / path).read_bytes()), expected)
        results = {result["id"]: result for result in receipt["results"]}
        self.assertGreaterEqual(len(results), 16)
        self.assertEqual({result["status"] for result in results.values()}, {"passed"})
        self.assertTrue({"sequential", "balanced", "reordered-repeated", "missing-input"} <= results.keys())
        executions = receipt["executions"]
        self.assertEqual(len(executions), 14)
        self.assertEqual(sum(execution["stage"] == "execute" for execution in executions), 2)
        for execution in executions:
            if "compile_fail/" in execution["source"]:
                self.assertNotEqual(execution["exit_code"], 0)
                self.assertTrue(execution["diagnostic"])
            else:
                self.assertEqual(execution["exit_code"], 0)
        rejected = next(execution for execution in executions
                        if execution["source"] == "tests/compile_fail/reduction_plan_missing_input.mojo")
        self.assertEqual(rejected["diagnostic"], "argmax reduction plan rejected")


if __name__ == "__main__":
    unittest.main()
