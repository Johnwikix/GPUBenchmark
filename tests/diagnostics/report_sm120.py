"""Audit real GPU results and correlate only formal sample windows with SMI."""
import csv
import hashlib
import html
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/sm120-acc16'
rows = []

def number(s):
    return float(s.strip().split()[0])

for command_file in sorted(OUT.glob('*/command.json')):
    directory = command_file.parent
    command = json.loads(command_file.read_text(encoding='utf-8-sig'))
    result_path = directory / 'result.json'
    if not result_path.exists():
        continue
    doc = json.loads(result_path.read_text(encoding='utf-8-sig'))
    telemetry = []
    with (directory / 'gpu-telemetry.csv').open(encoding='utf-8-sig') as stream:
        for raw in csv.DictReader(stream):
            sample = {k.strip(): (v.strip() if v else '') for k, v in raw.items() if k}
            try:
                stamp = datetime.strptime(sample['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone(timedelta(minutes=command.get('utc_offset_minutes', 480))))
                sample['unix_ms'] = stamp.timestamp() * 1000
                telemetry.append(sample)
            except (ValueError, KeyError):
                continue
    for r in doc['results']:
        valid = r['result_status'] == 'ok' and r['validation_status'] == 'passed'
        row = {'directory': directory.name, 'source': result_path.relative_to(OUT).as_posix(), 'case_id': r['case_id'], 'status': r['result_status'], 'validation': r['validation_status'], 'reason': r['reason_message'], 'median_tflops': r['median_tflops'], 'noisy': r.get('noisy'), 'shape': r['configuration'].get('logical_shape'), 'library': r['configuration'].get('library'), 'kernel': r['configuration'].get('kernel_variant'), 'input_generator': r['configuration'].get('input_generator'), 'command': command, 'telemetry': None}
        if valid:
            assert len(r['raw_samples']) >= 7 and r['measurement_duration'] >= 1000
            assert r['configuration']['actual_warmup_ms'] >= 500
            if r['suite'] == 'gemm':
                shape = row['shape']
                for s in r['raw_samples']:
                    assert s['flops'] == 2 * shape['m'] * shape['n'] * shape['k'] * s['repetitions']
                assert r['output'] == r['accumulation'] and r['density'] == 'dense'
                assert r['error_metrics']['post_measurement']
                if r['case_id'] == 'gemm.fp16.acc16':
                    assert r['accumulation'] == 'fp16'
                    assert r['tolerance_policy']['id'] == 'acc16-signed-dyadic-exact-v1'
                    assert all(c['max_absolute_error'] == 0 for c in r['error_metrics']['pre_measurement'] + r['error_metrics']['post_measurement'])
                if r['input_a'] == 'ocp_e4m3':
                    assert r['accumulation'] == 'fp32' and not r['configuration']['fast_accum']
                    assert all(c['scaled_reference_observable'] for c in r['error_metrics']['pre_measurement'] + r['error_metrics']['post_measurement'])
            times = [s['elapsed_ms'] for s in r['raw_samples']]
            calculated = r['raw_samples'][0]['flops'] / statistics.median(times) / 1e9
            assert math.isclose(calculated, r['median_tflops'], rel_tol=1e-10)
            windows = [(s['start_unix_ms'], s['end_unix_ms']) for s in r['raw_samples'] if 'start_unix_ms' in s]
            observed = [t for t in telemetry if any(a <= t['unix_ms'] <= b for a, b in windows)]
            if observed:
                expected_uuid=r['device'].get('identity',{}).get('uuid','').lower().replace('-','')
                assert all(t['uuid'].lower().replace('gpu-','').replace('-','') == expected_uuid for t in observed)
                metrics = {'count': len(observed), 'scope': 'formal_sample_windows_only'}
                for label, field in [('gpu_busy_percent', 'utilization.gpu [%]'), ('memory_busy_percent', 'utilization.memory [%]'), ('sm_mhz', 'clocks.current.sm [MHz]'), ('power_watts', 'power.draw [W]'), ('temperature_c', 'temperature.gpu')]:
                    values = [number(t[field]) for t in observed]
                    metrics[label] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
                for reason in ('clocks_event_reasons.sw_power_cap', 'clocks_event_reasons.hw_slowdown'):
                    metrics[reason] = sum(t.get(reason) == 'Active' for t in observed) / len(observed)
                row['telemetry'] = metrics
        else:
            assert r['median_tflops'] is None
        rows.append(row)

summary = {'device': 'RTX 5070 Ti', 'observations': rows, 'limitations': ['GPU busy is not Tensor pipeline occupancy.', 'SMI utilization is driver-window averaged; telemetry is polled at100ms.', 'acc16 has a separate signed dyadic peak input contract and exact validation.', 'No unsuccessful sample is promoted to a score.']}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RTX5070Ti SM120 / acc16 实测</title><style>body{font:15px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px;text-align:left}pre{white-space:pre-wrap}a{color:#125b91}</style><h1>RTX 5070 Ti：SM120 FP8 与独立 FP16 acc16</h1><p>原始命令、版本、样本、数值校验与100ms GPU遥测均保留。FP16 acc32保留；acc16使用独立有符号dyadic峰值输入，严格零误差校验。FP8保持E4M3、FP32累加/输出、dense和非单位标量缩放。</p><p>表中GPU忙碌率、SM频率、功耗、温度均只统计正式样本UTC时间窗口；无时间对齐证据时留空。忙碌率不是Tensor管线占用率；功耗为驱动采样读数，瞬时值可高于设定上限。</p><table><tr><th>运行 / case</th><th>M×N×K / 库</th><th>Median TFLOP/s</th><th>GPU忙碌率中位数</th><th>SM MHz中位数</th><th>功耗W中位数 / 温度°C</th><th>功耗限制活动比例</th><th>校验 / 状态</th></tr>']
def metric(t, key):
    return f'{t[key]["median"]:.1f}' if t else '—'
for r in rows:
    t = r['telemetry']; s = r['shape']; shape = '×'.join(str(s[k]) for k in ('m', 'n', 'k')) if s else 'ALU'
    parts.append('<tr><td><a href="' + html.escape(r['source']) + '">' + html.escape(r['directory'] + ' / ' + r['case_id']) + '</a></td><td>' + shape + ' / ' + html.escape(str(r['library'])) + '</td><td>' + (f'{r["median_tflops"]:.3f}' if r['median_tflops'] else '—') + '</td><td>' + metric(t, 'gpu_busy_percent') + '</td><td>' + metric(t, 'sm_mhz') + '</td><td>' + metric(t, 'power_watts') + ' / ' + metric(t, 'temperature_c') + '</td><td>' + (f'{100*t["clocks_event_reasons.sw_power_cap"]:.0f}%' if t else '—') + '</td><td>' + html.escape(r['validation'] + ' / ' + r['status'] + ' ' + r['reason']) + '</td></tr>')
parts.append('</table><h2>实际命令</h2>')
for c in {r['directory']: r['command'] for r in rows}.values():
    parts.append('<pre>' + html.escape(c['executable'] + ' ' + ' '.join('"' + a + '"' for a in c['arguments'])) + '\nexit=' + str(c['exit_code']) + '</pre>')
parts.append('<p><a href="summary.json">独立2MNKR审计与遥测统计</a>。失败、SKIP与噪声结果保留；不同形状、精度、库和输入生成器不混为单一排名。所有有效成绩来自本机真实GPU；编译和调度fixture不生成性能成绩。硬件性能计数器权限仍不足，未据SMI忙碌率宣称Tensor满载。</p></html>')
(OUT / 'REPORT.html').write_text('\n'.join(parts), encoding='utf-8')
print(f'Audited {len(rows)} real GPU result rows; successful={sum(r["status"] == "ok" for r in rows)}')
