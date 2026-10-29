"""Actual Windows console cancellation check; never invents GPU scores."""
import ctypes
import json
import signal
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
evidence = root / "results" / "console-cancel" / str(time.time_ns())
evidence.mkdir(parents=True)
output = evidence / "result.json"
args = [str(root / "out/Release/gpubench.exe"), "run", "--backend", "cuda",
        "--device", "nvidia:0", "--suite", "alu", "--dtype", "fp32,fp16",
        "--json", str(output), "--log-dir", str(evidence / "workers")]
startup = subprocess.STARTUPINFO()
startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startup.wShowWindow = subprocess.SW_HIDE
started = time.time()
record = {"command": args, "mechanism": "Windows CTRL_BREAK_EVENT to hidden child console; same host handler as Ctrl+C"}
with (evidence / "stdout.log").open("w", encoding="utf-8") as out, (evidence / "stderr.log").open("w", encoding="utf-8") as err:
    process = subprocess.Popen(args, stdout=out, stderr=err, startupinfo=startup,
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
    try:
        deadline = time.monotonic() + 30
        first_completed = False
        while process.poll() is None and time.monotonic() < deadline:
            if output.exists():
                try:
                    doc = json.loads(output.read_text(encoding="utf-8"))
                    first_completed = any(r["result_status"] == "ok" for r in doc["results"])
                    if first_completed:
                        break
                except (OSError, ValueError):
                    pass
            time.sleep(.02)
        if not first_completed:
            raise RuntimeError("first actual GPU result did not complete in 30 seconds")
        time.sleep(.1)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        kernel.FreeConsole()
        if not kernel.AttachConsole(process.pid):
            raise ctypes.WinError(ctypes.get_last_error())
        # AttachConsole resets the Win32 control-handler table. Protect the
        # helper after attaching, before broadcasting to the child's console.
        ignore_break = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(lambda _: 1)
        if not kernel.SetConsoleCtrlHandler(ignore_break, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        sent_at = time.monotonic()
        if not kernel.GenerateConsoleCtrlEvent(1, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(.05)
        kernel.FreeConsole()
        code = process.wait(timeout=8)
        record.update(exit_code=code, cancel_latency_ms=(time.monotonic()-sent_at)*1000)
        final = json.loads(output.read_text(encoding="utf-8"))
        record["completed_preserved"] = any(r["result_status"] == "ok" for r in final["results"])
        record["passed"] = code == 130 and final["summary"]["exit_code"] == 130 and record["completed_preserved"]
    except Exception as error:
        record.update(passed=False, error=str(error))
        if process.poll() is None:
            process.terminate()  # Job Object closes and stops its worker.
            process.wait(timeout=8)
record["wall_seconds"] = time.time()-started
(evidence / "check.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(json.dumps(record))
raise SystemExit(0 if record.get("passed") else 1)
