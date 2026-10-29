"""Independent FLOP/time audit plus exact launched-function ISA evidence."""
import hashlib
import json
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "results/throughput-diagnosis"
data = json.loads((root / "repro-4096.json").read_text(encoding="utf-8-sig"))
audit = []
for result in data["results"]:
    config, samples = result["configuration"], result["raw_samples"]
    repetitions = samples[0]["repetitions"]
    expected = 2*config["m"]*config["n"]*config["k"]*repetitions
    assert all(s["flops"] == expected and s["repetitions"] == repetitions for s in samples)
    elapsed = statistics.median(s["elapsed_ms"] for s in samples)
    throughput = expected/(elapsed*1e9)
    assert abs(throughput-result["median_tflops"]) < 1e-10*throughput
    audit.append({"case": result["case_id"], "expected_batch_flops": expected,
                  "recorded_batch_flops": samples[0]["flops"],
                  "recomputed_tflops": throughput, "reported_tflops": result["median_tflops"],
                  "wall_over_device": statistics.median(s["wall_ms"] for s in samples)/elapsed})
(root / "count-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
evidence = []
for dtype in ("fp32", "fp16", "fp8"):
    events = [json.loads(line) for line in (root/f"{dtype}-kernel-cubin.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    kernel = next(event["actual_kernel"] for event in events if "actual_kernel" in event)
    found = False
    for source in root.glob(f"{dtype}-module-*.cubin.sass.txt"):
        text = source.read_text(encoding="utf-8-sig")
        match = re.search(r"Function\s*:\s*"+re.escape(kernel)+r"(?=\s)", text)
        if not match:
            continue
        after = re.search(r"Function\s*:", text[match.end():])
        end = match.end()+after.start() if after else len(text)
        selected = text[match.start():end]
        output = root/f"{dtype}-selected-kernel.sass.txt"
        output.write_text(selected, encoding="utf-8")
        families = sorted(set(re.findall(r"\b(?:QMMA|HMMA|FFMA)[.A-Z0-9]*", selected)))
        assert any(name.startswith({"fp32":"FFMA", "fp16":"HMMA", "fp8":"QMMA"}[dtype]) for name in families)
        evidence.append({"dtype": dtype, "actual_kernel": kernel,
                         "actual_launch_observed": True, "source": str(source),
                         "selected_sass": str(output), "selected_sass_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                         "instruction_families": families})
        found = True
        break
    assert found, f"launched kernel missing from captured modules: {kernel}"
(root/"actual-kernel-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print("Independent counting/timing checks passed; launched FP16 HMMA and FP8 QMMA confirmed.")
