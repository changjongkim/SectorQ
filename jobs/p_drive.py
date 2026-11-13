"""The paulistrings-rs baseline (jobs/p_ki.py) inside a four-node allocation: every way the library can be placed on
the nodes, each down the same ladder of cutoffs until a run fails or exceeds 40 minutes.

  env/bin/python3 jobs/p_drive.py --count          # runs still open
  env/bin/python3 jobs/p_drive.py --job <job id>   # work through the open runs in that allocation
  env/bin/python3 jobs/p_drive.py --table          # what is recorded

Records: queue_p/done/<id>.json, logs: logs/p/<id>.out. A ladder stops at its first run that is not ok."""
import argparse
import glob
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q, LOGS = os.path.join(ROOT, "queue_p", "done"), os.path.join(ROOT, "logs", "p")
PY = os.path.join(ROOT, "third_party", "paulistrings-rs", ".venv", "bin", "python")
# 128 and 256 ranks: the library allows at most 2^P_MAX_BITS = 64 partitions. The copy paulistrings-rs-p8 raises
# P_MAX_BITS from 6 to 8 and widens the in-process transport's 64-bit rank mask (uncommitted diff in that tree;
# rows for 1-6 bits are unchanged). Used for mg128, mg256 and the check run p.p8chk.* (same point as p.mg64.ki.5e-6
# with the new build, which a newer Rust compiler built).
PY8 = os.path.join(ROOT, "third_party", "paulistrings-rs-p8", ".venv", "bin", "python")
P8_MODES = ("mg128", "mg256")
P8_CHECK = ("p.p8chk.mg64.ki.5e-6", "mg64", "ki", "5e-6")
CUTS = ("1e-3", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6")
SOME = ("g1", "g4", "mg16", "m16", "mg32", "mg64")   # the other six circuits: one GPU, one node, 4/8/16 nodes on GPUs, four nodes on the host
# circuit: (--case of jobs/p_ki.py, cutoffs of its ladder, modes); kicked Ising uses the library's own builder
CIRCUITS = {
    "ki": ("", CUTS, None),
    "tfim": ("tfim", ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), SOME),
    "noise": ("noise", ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), SOME),
    "tfimsq": ("tfimsq", ("1e-4", "3e-5", "1e-5", "3e-6"), SOME),
    "xxz": ("xxz", ("1e-4", "3e-5", "1e-5", "3e-6"), SOME),
    "qaoa": ("qaoa", ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), SOME),
    "hea": ("hea", ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7"), SOME),
}
BIG = ("mg32", "mg64")
MORE = {"ki": "5e-7", "tfim": "5e-7", "noise": "5e-7", "tfimsq": "1e-6", "xxz": "1e-6", "qaoa": "5e-7", "hea": "1e-7"}   # one step further on 32 and 64 GPUs
LIMIT = 2400
ONE = "-N 1 -n 1 -c 128 --cpu-bind=none"
ALL = "-N 4 -n 16 --ntasks-per-node=4 -c 32 --cpu-bind=cores"
ALL8 = "-N 8 -n 32 --ntasks-per-node=4 -c 32 --cpu-bind=cores"      # batch jobs (jobs/p_batch.sh)
ALL16 = "-N 16 -n 64 --ntasks-per-node=4 -c 32 --cpu-bind=cores"
ALL2 = "-N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores"
ALL32 = "-N 32 -n 128 --ntasks-per-node=4 -c 32 --cpu-bind=cores"
ALL64 = "-N 64 -n 256 --ntasks-per-node=4 -c 32 --cpu-bind=cores"
# name: (what it is, srun placement, arguments of jobs/p_ki.py)
MODES = {
    "h1": ("one node, host, one partition per NUMA domain, seeded rows", ONE, "--partitions auto"),
    "h1c": ("one node, host, one partition per NUMA domain, cut rows", ONE, "--partitions auto --rows cut"),
    "h1u": ("one node, host, unpartitioned", ONE, ""),
    "g1": ("one GPU", ONE + " --gpus-per-node=4", "--device 0"),
    "g4": ("one node, 4 GPUs, seeded rows", ONE + " --gpus-per-node=4", "--device 0,1,2,3"),
    "g4c": ("one node, 4 GPUs, cut rows", ONE + " --gpus-per-node=4", "--device 0,1,2,3 --rows cut"),
    "m16": ("4 nodes, 16 MPI ranks on the host, seeded rows", ALL, "--mpi 1"),
    "m16c": ("4 nodes, 16 MPI ranks on the host, cut rows", ALL, "--mpi 1 --rows cut"),
    "mg16": ("4 nodes, 16 MPI ranks with one GPU each, seeded rows", ALL + " --gpus-per-node=4", "--mpi 1 --device auto"),
    "mg16c": ("4 nodes, 16 MPI ranks with one GPU each, cut rows", ALL + " --gpus-per-node=4", "--mpi 1 --device auto --rows cut"),
    "mg32": ("8 nodes, 32 MPI ranks with one GPU each, seeded rows", ALL8 + " --gpus-per-node=4", "--mpi 1 --device auto"),
    "mg64": ("16 nodes, 64 MPI ranks with one GPU each, seeded rows", ALL16 + " --gpus-per-node=4", "--mpi 1 --device auto"),
    "g2": ("one node, 2 GPUs, seeded rows", ONE + " --gpus-per-node=4", "--device 0,1"),
    "mg8": ("2 nodes, 8 MPI ranks with one GPU each, seeded rows", ALL2 + " --gpus-per-node=4", "--mpi 1 --device auto"),
    "mg128": ("32 nodes, 128 MPI ranks with one GPU each, seeded rows", ALL32 + " --gpus-per-node=4", "--mpi 1 --device auto"),
    "mg256": ("64 nodes, 256 MPI ranks with one GPU each, seeded rows", ALL64 + " --gpus-per-node=4", "--mpi 1 --device auto"),
}
# NCCL carries the exchange between the GPUs of different ranks; the library binds one version of its interface, so
# the versions installed here are tried in this order and the first that runs is kept (queue_p/nccl).
NCCLS = ("2.24.3", "2.21.5", "2.29.2-cu13", "2.19.4", "2.18.3-cu12")
ENV = ("module load nccl/{nccl} >/dev/null 2>&1; export MPICH_GPU_SUPPORT_ENABLED=0; "
       "export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.4/lib64:$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64; ")


# 2026-11-09: paulistrings-rs joins the main figures with its own truncation rule (after every operation), as MonoProp
# with its own. Its ladders get the cutoffs of those figures: the single node (Figure 8: one and four GPUs), the
# strong-scaling panels (kicked Ising on 16-256 GPUs), the seven circuits on 16 and 128 GPUs, and weak scaling
# (WEAK below). A ladder still stops at its first run that is not ok; the cutoffs are sorted from the largest down.
BIG3 = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7")
EXTRA = {
    ("ki", "g1"): ("5e-4", "3e-4", "2e-4", "5e-5", "2e-5", "5e-6", "2e-6"),
    ("ki", "g4"): ("5e-4", "3e-4", "2e-4", "5e-5", "2e-5", "5e-6", "2e-6"),
    ("ki", "mg16"): ("5e-6", "2e-6"), ("tfim", "mg16"): ("5e-6", "2e-6"), ("noise", "mg16"): ("5e-6", "2e-6"),
    ("ki", "mg32"): ("5e-6", "2e-6"), ("ki", "mg64"): ("5e-6", "2e-6"),
    ("tfimsq", "mg16"): ("5e-5", "2e-5", "5e-6", "2e-6", "1e-6", "5e-7"),
    ("xxz", "mg16"): ("5e-5", "2e-5", "5e-6", "2e-6", "1e-6", "5e-7"),
    ("qaoa", "mg16"): ("5e-5", "2e-5", "5e-6", "2e-6", "5e-7", "3e-7"),
    ("hea", "mg16"): ("5e-5", "2e-5", "5e-6", "2e-6", "5e-7", "1e-7"),
}
LADDER128 = {"ki": BIG3, "tfim": BIG3, "noise": BIG3,
             "tfimsq": BIG3, "xxz": BIG3,
             "qaoa": ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "5e-7", "3e-7", "2e-7", "1e-7", "7e-8", "5e-8"),
             "hea": ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "5e-7", "3e-7", "2e-7", "1e-7", "7e-8", "5e-8", "3e-8")}
LADDER256 = {"ki": BIG3}           # 3e-7 and 2e-7 added 2026-11-12 (review: below its ok run at 5e-7)
# weak scaling: the cutoffs of make_eval_figures.WEAK_S/WEAK_G and BIG_WEAK_S/BIG_WEAK (512 GPUs are not run)
WEAK_MODE = {1: "g1", 2: "g2", 4: "g4", 8: "mg8", 16: "mg16", 32: "mg32", 64: "mg64", 128: "mg128", 256: "mg256"}
WEAK = {"S": {1: "6.2e-5", 2: "4.3e-5", 4: "3e-5", 8: "2.1e-5", 16: "1.5e-5", 32: "1e-5", 64: "7.1e-6", 128: "4.9e-6", 256: "3.5e-6"},
        "L": {1: "2e-5", 2: "1.4e-5", 4: "1e-5", 8: "7e-6", 16: "4.8e-6", 32: "3.4e-6", 64: "2.3e-6", 128: "1.6e-6", 256: "1.15e-6"}}


def cuts_of(c, m):
    """The ladder of one circuit in one mode: on 32 and 64 GPUs one cutoff below the ladder of the four-node runs, the
    cutoffs of the main figures (EXTRA), and the ladders on 128 and 256 GPUs."""
    if m == "mg128":
        return LADDER128[c]
    if m == "mg256":
        return LADDER256.get(c, ())
    if m not in (CIRCUITS[c][2] or MODES) or m in ("g2", "mg8"):
        return ()
    cuts = set(CIRCUITS[c][1] + ((MORE[c],) if m in BIG else ()) + EXTRA.get((c, m), ()))
    return tuple(sorted(cuts, key=lambda x: -float(x)))


def runs():
    out = [(f"p.{m}.{c}.{x}", m, c, x) for c in CIRCUITS for m in MODES for x in cuts_of(c, m)]
    have = {(m, c, x) for _, m, c, x in out}
    for load, cut in WEAK.items():
        for c in ("ki", "tfim", "noise"):
            for g, m in WEAK_MODE.items():
                if (m, c, cut[g]) not in have:             # a point of the sweep already measures it
                    out.append((f"p.weak{load}.{m}.{c}.{cut[g]}", m, c, cut[g]))
    out.insert(0, P8_CHECK)
    return out


def record(rid):
    try:
        return json.load(open(os.path.join(Q, rid + ".json")))
    except (OSError, ValueError):
        return None


_LEFT = {}


def left(job):
    """Seconds left; squeue that does not answer is asked again, then the count goes on from its last answer (an empty
    answer read as 0 s ended the scale jobs 59768999 and 59769000 early)."""
    for attempt in range(4):
        try:
            out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True, timeout=60).stdout.strip()
        except subprocess.TimeoutExpired:
            out = ""
        m = re.match(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", out)
        if m:
            s = int(m.group(1) or 0) * 86400 + int(m.group(2) or 0) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            _LEFT[job] = (s, time.time())
            return s
        if attempt < 3:
            time.sleep(15)
    if job in _LEFT:
        s, at = _LEFT[job]
        return max(0, s - int(time.time() - at))
    return 0


def launch(rid, m, c, x, job, host, nccl):
    """One run with one NCCL version; returns its record."""
    what, place, args = MODES[m]
    where = host if place.startswith("-N 1 ") else ""
    py = PY8 if (m in P8_MODES or rid.startswith("p.p8chk.")) else PY
    # communicator setup over 128/256 ranks: a longer bound on each wire wait than the library's 300 s
    wait = "export PAULISTRINGS_NCCL_TIMEOUT_S=900; " if m in P8_MODES else ""
    cmd = ENV.replace("{nccl}", nccl) + wait + f"srun --jobid={job} --overlap {place}{where} --kill-on-bad-exit=1 {py} {ROOT}/jobs/p_ki.py --cutoff {x} {args}" + (f" --case {CIRCUITS[c][0]}" if CIRCUITS[c][0] else "")
    log = os.path.join(LOGS, rid + ".out")
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"===== {rid}: {what}\n{cmd}\n")
        f.flush()
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        code, status, seen = None, None, None
        while code is None:
            try:
                code = p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if seen is None and "\nRESULT " in open(log, errors="replace").read():
                    seen = time.time()          # the result is printed; a run that then does not exit is stopped after a minute
                over = time.time() - t0 > LIMIT
                if over or (seen is not None and time.time() - seen > 60):
                    subprocess.run(["bash", "-c", f"kill -TERM -{p.pid}"], capture_output=True)
                    time.sleep(15)
                    subprocess.run(["bash", "-c", f"kill -KILL -{p.pid}"], capture_output=True)
                    code, status = -1, ("tl" if seen is None else None)
    text = open(log, errors="replace").read()
    rec = dict(status=status or "fail", exit=code, seconds=round(time.time() - t0, 1), log=f"logs/p/{rid}.out", job=job,
               time=time.strftime("%m-%dT%H:%M:%S"), mode=m, what=what)
    hit = re.findall(r"^RESULT (\{.*\})$", text, re.M)
    if hit and status is None:
        rec.update(json.loads(hit[-1]))
        rec.update(status="ok", left_running=code != 0)
        if m.startswith("mg"):
            rec["nccl"] = nccl
    elif status is None and re.search(r"out of memory|OutOfMemory|oom-kill|oom_kill|CUDA_ERROR_OUT_OF_MEMORY|Killed|bad_alloc|MemoryError", text, re.I):
        rec["status"] = "oom"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--modes", default="", help="only these modes (comma separated)")
    ap.add_argument("--circuits", default="", help="only these circuits (comma separated)")
    ap.add_argument("--node", type=int, default=-1, help="one-node runs on this node of the allocation (0-3), so that four ladders run side by side")
    a = ap.parse_args()
    os.makedirs(Q, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    if a.count:
        print(sum(1 for rid, m, c, x in runs() if record(rid) is None))
        return
    if a.table:
        short = {"skipped_after_failure": "-"}
        for c, (_, cuts, modes) in CIRCUITS.items():
            print(c)
            for m in (modes or MODES):
                cells = []
                for x in cuts_of(c, m):
                    r = record(f"p.{m}.{c}.{x}") or {}
                    cells.append(f"{x}: " + (f"{r['wall']:.1f}s" if r.get("status") == "ok" else short.get(r.get("status"), r.get("status", "open"))))
                print(f"  {m:6s} {MODES[m][0]:58s} " + "  ".join(cells))
        return
    only = [m for m in a.modes.split(",") if m]
    which = [c for c in a.circuits.split(",") if c]
    host = ""
    if a.node >= 0:
        nodes = subprocess.run(["bash", "-c", f"scontrol show hostnames $(squeue -h -j {a.job} -o %N)"], capture_output=True, text=True).stdout.split()
        host = f" --nodelist={nodes[a.node]}"
    kept = os.path.join(os.path.dirname(Q), "nccl")
    have = int(subprocess.run(["squeue", "-h", "-j", a.job, "-o", "%D"], capture_output=True, text=True).stdout.strip() or 0)
    for rid, m, c, x in runs():
        if record(rid) is not None or (only and m not in only) or (which and c not in which):
            continue
        if int(re.match(r"-N (\d+)", MODES[m][1]).group(1)) > have:
            continue                                    # needs more nodes than this allocation has
        cuts = cuts_of(c, m)
        before = [] if rid.startswith("p.weak") else [record(f"p.{m}.{c}.{y}") for y in cuts[:cuts.index(x)]]
        if any(b is not None and b.get("status") != "ok" for b in before):
            json.dump(dict(status="skipped_after_failure"), open(os.path.join(Q, rid + ".json"), "w"))
            continue
        if left(a.job) < LIMIT + 180:
            print(f"{time.strftime('%m-%dT%H:%M:%S')} not enough time left in the allocation for {rid}", flush=True)
            return
        # the first run with one GPU per rank tries the NCCL versions in turn; the one that runs is kept for the rest
        tries = [open(kept).read().strip()] if os.path.exists(kept) else list(NCCLS if m.startswith("mg") else NCCLS[:1])
        for nccl in tries:
            rec = launch(rid, m, c, x, a.job, host, nccl)
            print(f"{rec['time']} {rid}: {rec['status']} wall={rec.get('wall')} expectation={rec.get('expectation')} ({rec['seconds']} s)"
                  + (f" nccl {nccl}" if m.startswith("mg") else ""), flush=True)
            if rec["status"] != "fail":
                break
        if rec["status"] == "ok" and m.startswith("mg"):
            open(kept, "w").write(nccl)
        json.dump(rec, open(os.path.join(Q, rid + ".json"), "w"))


if __name__ == "__main__":
    main()
