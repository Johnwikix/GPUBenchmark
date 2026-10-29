"""Independently audit recorded GPU samples and render the optimization report."""
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/throughput-optimization"


def audit(r, ops=None):
    assert r["result_status"] == "ok" and r["validation_status"] == "passed"
    samples = r["raw_samples"]
    assert len(samples) >= 7 and sum(s["elapsed_ms"] for s in samples) >= 1000
    assert r["configuration"]["actual_warmup_ms"] >= 500
    for s in samples:
        assert s["elapsed_ms"] > 0 and s["wall_ms"] > 0
        if ops is not None:
            assert s["flops"] == ops * s["repetitions"]
    calculated = samples[0]["flops"] / statistics.median(s["elapsed_ms"] for s in samples) / 1e9
    assert math.isclose(calculated, r["median_tflops"], rel_tol=1e-10)
    return {"median_tflops": calculated, "samples": len(samples), "sample_flops_audited": ops is not None}


rows = []
for path in sorted(OUT.glob("fp*.json")):
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    assert doc["diagnostic_only"] and doc.get("post_validation")
    for check in (doc["validation"], doc["post_validation"]):
        assert check["sampled_elements"] == 128
        if doc["dtype"] == "fp8":
            assert check["scaled_reference_observable"]
    row = {"file": path.relative_to(OUT).as_posix(), "kind": "diagnostic", "dtype": doc["dtype"], "size": doc["size"], "variant": doc["variant"], **audit(doc["measurement"], 2 * doc["size"] ** 3)}
    if "extended_search" in doc:
        ex = doc["extended_search"]
        row["search"] = {"checked": ex["checked"], "timed": len(ex["timed"]), "budget_exhausted": ex["time_budget_exhausted"]}
    rows.append(row)

commands = json.loads((OUT / "production/commands.json").read_text(encoding="utf-8-sig"))
assert len(commands) == 4 and all(c["exit_code"] == 0 for c in commands)
for path in sorted((OUT / "production").glob("*.json")):
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or "results" not in doc:
        continue
    for r in doc["results"]:
        ops, shape = None, None
        if r["suite"] == "gemm":
            shape = r["configuration"]["logical_shape"]
            ops = 2 * shape["m"] * shape["n"] * shape["k"]
            assert r["density"] == "dense" and r["accumulation"] == r["output"] == "fp32"
            assert not r["configuration"]["fast_accum"]
            assert r["error_metrics"]["post_measurement"]
        rows.append({"file": path.relative_to(OUT).as_posix(), "kind": "production", "case": r["case_id"], "shape": shape, **audit(r, ops)})

summary = {"status": "passed", "device": "RTX 5070 Ti", "results": rows, "production_commands": commands, "limits": "FP8 450 not reached; no validated acc16 result; external benchmark original binaries not reproduced; hardware counters denied. Diagnostics are distinct from production certification."}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RTX 5070 Ti 优化复测</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px;text-align:left}pre{white-space:pre-wrap}td:last-child{font-variant-numeric:tabular-nums}</style><h1>RTX 5070 Ti：FP16 / FP8 优化复测</h1><p>Windows 原生 CUDA 13.2.2 / cuBLASLt 13.4.1.3 / 驱动 617.14。2026-10-21。正式 worker 与诊断记录分列。仅展示真实 GPU、前后校验通过的数据；全部原始样本与命令在链接文件内。</p><p>源码对照：部分工具的 FP16 使用 half 累加，FP8 使用不同输出或 FAST_ACCUM。GEMM 统一按 2MNK 计数，未发现应当乘二的证据。本轮未达到 FP16 220 / FP8 450；不能混用累加精度或微基准分数。</p>']
for kind, label in (("production", "正式 host / worker 回归"), ("diagnostic", "独立诊断对照")):
    parts.append(f'<h2>{label}</h2><table><tr><th>来源与case</th><th>尺寸 / 变体</th><th>Median TFLOP/s</th></tr>')
    for r in rows:
        if r["kind"] != kind:
            continue
        name = r.get("case", r.get("dtype"))
        description = str(r.get("shape") or r.get("size", "ALU")) + " " + r.get("variant", "")
        parts.append(f'<tr><td><a href="{html.escape(r["file"])}">{html.escape(name)}</a></td><td>{html.escape(description)}</td><td>{r["median_tflops"]:.3f}</td></tr>')
    parts.append('</table>')
parts.append('<h2>实际正式命令</h2>')
for c in commands:
    parts.append('<pre>' + html.escape(c["executable"] + " " + " ".join('"' + a + '"' for a in c["arguments"])) + "\nexit=" + str(c["exit_code"]) + '</pre>')
parts.append('<p>可复核 <a href="summary.json">独立计数审计</a>、<a href="production/environment.json">二进制/库哈希</a>、<a href="production/gpu-before.txt">GPU环境</a>。选优预热和调优时间独立于正式计时；数据准备不计入；同stream事件覆盖完整batch。计数器权限仍不足，未宣称完成瓶颈归因。</p></html>')
(OUT / "REPORT.html").write_text("\n".join(parts), encoding="utf-8")
manifest = [{"path": p.relative_to(OUT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "SHA256.json"]
(OUT / "SHA256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Audited {len(rows)} validated GPU results; production commands: {len(commands)}")
