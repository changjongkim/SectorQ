#!/usr/bin/env python3
"""Campaign g runner: executes the runs of queue_g/todo*.jsonl that need as many nodes as hosts given.

  python3 jobs/g_run.py --job <allocation id> --hosts <host[,host...]>
  python3 jobs/g_run.py --count            number of runs still open

Several runners (one per node, per pair of nodes, per allocation) work on the same lists: a run is claimed with
mkdir queue_g/lock/<id>; a finished run leaves queue_g/done/<id> (JSON: status, wall, expectation) and
logs/g/<id>.out. A ladder (runs with the same "ladder" value, in the order of the list) stops at its first failure:
the later runs are marked skipped_after_failure. A step that the earlier steps predict to take more than 1.2 times
the limit is marked predicted_tl and not run. Statuses: ok, oom, tl, fail, skipped_after_failure, predicted_tl.
"""
import argparse
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
LOGS = os.path.join(ROOT, "logs", "g")
BIN = os.environ.get("PP_BIN", "")
DROP = re.compile(r"^$|^srun: (Job|Step)|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called")


def load_items():
    items = []
    for path in sorted(glob.glob(os.path.join(Q, "todo*.jsonl"))):
        for line in open(path):
            if line.strip():
                items.append(json.loads(line))
    return items


def status_of(item_id):
    p = os.path.join(Q, "done", item_id)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except ValueError:                       # being written
        return None


def mark(item_id, **record):
    record["time"] = time.strftime("%m-%dT%H:%M:%S")
    tmp = os.path.join(Q, "done", "." + item_id + ".tmp")
    json.dump(record, open(tmp, "w"))
    os.replace(tmp, os.path.join(Q, "done", item_id))
    with open(os.path.join(Q, "runner.log"), "a") as f:
        f.write(json.dumps(dict(record, id=item_id)) + "\n")


def squeue(job, fmt, state=None):
    cmd = ["squeue", "-h", "-j", str(job), "-o", fmt] + (["-t", state] if state else [])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


_LEFT = {}


def left(job):
    """Seconds left in the allocation. squeue can fail to answer for a while; jobs 59768999 (64 nodes, 3 h) and
    59769000 (128 nodes, 4.5 h) ended after about 40 minutes with runs still open, which is what one empty answer
    did here before (0 s left = the allocation is over). Ask again a few times; if squeue stays silent, count down
    from its last answer."""
    for attempt in range(4):
        t = squeue(job, "%L")
        if t and re.match(r"^[\d:-]+$", t):
            days = 0
            if "-" in t:
                d, t = t.split("-")
                days = int(d)
            parts = [int(x) for x in t.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0)
            s = days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
            _LEFT[job] = (s, time.time())
            return s
        if attempt < 3:
            time.sleep(15)
    if job in _LEFT:
        s, at = _LEFT[job]
        return max(0, s - int(time.time() - at))
    return 0


def clean_locks():
    """Claims of runs that did not finish and whose allocation is gone go back to the list."""
    for d in glob.glob(os.path.join(Q, "lock", "*")):
        item_id = os.path.basename(d)
        if os.path.exists(os.path.join(Q, "done", item_id)):
            continue
        try:
            owner = open(os.path.join(d, "owner")).read().split()[0]
        except (OSError, IndexError):
            if time.time() - os.path.getmtime(d) < 60:
                continue
            owner = ""
        if not owner or not squeue(owner, "%i", "RUNNING"):
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)
            try:
                os.rmdir(d)
            except OSError:
                pass


def predict(item, earlier):
    """Seconds the run is expected to take, from the earlier steps of its ladder (None: no basis)."""
    pts = [(e["coef"], s["wall"]) for e, s in earlier if s.get("status") == "ok" and s.get("wall") and e.get("coef")]
    if not pts or "coef" not in item or item["kind"] != "eng" and not item.get("predict"):
        return None
    c1, t1 = pts[-1]
    if t1 < 20 or item["coef"] >= c1:
        return None
    p = 1.0
    if len(pts) > 1 and pts[-2][1] > 5 and pts[-2][0] > c1:
        p = min(2.5, max(0.5, math.log(t1 / pts[-2][1]) / math.log(pts[-2][0] / c1)))
    return t1 * (c1 / item["coef"]) ** p


def binary_of(item):
    """SectorQ on more than one GPU runs the engine that chooses the rows of the next window while the window runs
    (bin/pp_engine_h, from 2026-10-10 01:20); SectorQ on one GPU and cuPauliProp run the first campaign binary
    (bin/pp_engine_g0), whose code for them is the same. bin/pp_engine_g makes the same choice for the runners
    that were started before. A run can name its binary itself ("bin"): the runs on more than 256 qubits and those with
    the total magnetization need bin/pp_engine_w (partition kernels for terms of any width, --obs-all)."""
    if BIN:
        return BIN
    if item.get("bin"):
        return item["bin"]
    return "bin/pp_engine_h" if item["kind"] == "eng" and item["nodes"] * item["per"] > 1 else "bin/pp_engine_g0"


def srun_prefix(job, hosts, tasks, per_node, cpus, bind):
    return ["srun", f"--jobid={job}", f"--nodelist={hosts}", "--overlap", "-N", str(len(hosts.split(","))), "-n", str(tasks),
            f"--ntasks-per-node={per_node}", "-c", str(cpus), "--gpus-per-node=4", f"--cpu-bind={bind}"]


def run_engine(item, job, hosts):
    nn = len(hosts.split(","))
    env = os.environ.copy()
    kind, per, args = item["kind"], item["per"], item["args"].split()
    binary = binary_of(item)
    if kind == "eng":
        env["MPICH_GPU_SUPPORT_ENABLED"] = "0"
        head = f"===== engine nodes={nn} gpus={nn * per} args: {item['args']} --gpus {per}"
        cmd = srun_prefix(job, hosts, nn, 1, 128, "none") + ["stdbuf", "-oL", "./" + binary, "--gpus", str(per)] + args
    elif kind == "ref":
        env["MPICH_GPU_SUPPORT_ENABLED"] = "0"
        head = f"===== ref nodes=1 gpus=1 args: {item['args']}"
        cmd = srun_prefix(job, hosts, 1, 1, 32, "none") + ["stdbuf", "-oL", "./" + binary, "--mode", "ref"] + args
    else:
        env["MPICH_GPU_SUPPORT_ENABLED"] = "1"
        env["CUPAULIPROP_COMM_LIB"] = os.path.join(ROOT, "bin", f"libcupp_mpi_{item.get('plugin', 'orig')}.so")
        head = f"===== cupp nodes={nn} gpus={nn * per} args: {item['args']} plugin={item.get('plugin', 'orig')}"
        cmd = srun_prefix(job, hosts, nn * per, per, 128 // per, "cores") + ["stdbuf", "-oL", "./" + binary, "--mode", "cupp"] + args
    sampler = None
    if item.get("smi"):
        sampler = subprocess.Popen(srun_prefix(job, hosts, nn, 1, 1, "none") +
                                   ["bash", "jobs/g_smi.sh", os.path.join(LOGS, "smi." + item["id"]), str(item["smi"])],
                                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2)
    log = os.path.join(LOGS, item["id"] + ".out")
    t0 = time.time()
    seen, lines = set(), 0
    with open(log, "w") as out:
        out.write(head + "\n" + "binary=" + os.path.basename(binary) + "\n")
        out.flush()
        proc = subprocess.Popen(["timeout", str(item["limit"])] + cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, errors="replace")
        for line in proc.stdout:
            line = line.rstrip("\n")
            if DROP.search(line):
                continue
            if kind != "eng":                 # the ranks repeat the same messages
                if line in seen or lines >= 80:
                    continue
                seen.add(line)
            lines += 1
            out.write(line + "\n")
            out.flush()
        code = proc.wait()
        out.write(f"exit={code}\n")
    seconds = time.time() - t0
    if sampler:
        try:
            os.killpg(sampler.pid, signal.SIGTERM)
            sampler.wait(timeout=30)
        except Exception:
            try:
                os.killpg(sampler.pid, signal.SIGKILL)
            except Exception:
                pass
    text = open(log).read()
    wall = re.search(r"^wall=([0-9.]+)", text, re.M)
    exp = re.search(r"^expectation=(\S+)", text, re.M)
    if code == 0 and (exp or "stopped at the checkpoint" in text or "--stop-at-ckpt" in item["args"]):
        status = "ok"
    elif code == 124:
        status = "tl"
    elif re.search(r"out of memory|distributed capacity|OUT_OF_MEMORY|bad_alloc|host allocation failed|Killed|oom-kill|Out Of Memory", text) or kind == "ref":
        status = "oom"
    else:
        status = "fail"
    return dict(status=status, exit=code, wall=float(wall.group(1)) if wall else None,
                expectation=float(exp.group(1)) if exp else None, seconds=round(seconds, 1), log=f"logs/g/{item['id']}.out",
                bin=os.path.basename(binary))


def run_item(item, job, hosts):
    if item["kind"] in ("eng", "ref", "cupp"):
        return run_engine(item, job, hosts)
    sys.path.insert(0, os.path.join(ROOT, "jobs"))
    import g_ext                              # external baselines
    return g_ext.run(item, job, hosts)


def supported(item):
    return item["kind"] in ("eng", "ref", "cupp") or os.path.exists(os.path.join(ROOT, "jobs", "g_ext.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--hosts")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated kinds to run (default: all)")
    a = ap.parse_args()
    if a.count:
        print(sum(1 for i in load_items() if not os.path.exists(os.path.join(Q, "done", i["id"]))))
        return
    nn = len(a.hosts.split(","))
    only = set(a.only.split(",")) - {""}
    idle = 0
    while True:
        items = [i for i in load_items()]
        by_ladder = {}
        for i in items:
            if i.get("ladder"):
                by_ladder.setdefault(i["ladder"], []).append(i)
        ran, blocked, unfit = False, 0, set()
        remaining = left(a.job)
        if remaining < 150:
            break
        for item in items:
            if item["nodes"] != nn or (only and item["kind"] not in only) or not supported(item):
                continue
            item_id = item["id"]
            if os.path.exists(os.path.join(Q, "done", item_id)):
                continue
            earlier = []
            if item.get("ladder"):
                ladder = by_ladder[item["ladder"]]
                earlier = [(e, status_of(e["id"])) for e in ladder[:[e["id"] for e in ladder].index(item_id)]]
            open_before = [e for e, s in earlier if s is None]
            if open_before:
                if any(os.path.isdir(os.path.join(Q, "lock", e["id"])) and e["id"] not in unfit for e in open_before):
                    blocked += 1               # another runner is on an earlier step
                unfit.add(item_id)
                continue
            if any(s["status"] != "ok" for _, s in earlier):
                mark(item_id, status="skipped_after_failure")
                continue
            expect = predict(item, earlier)
            if expect and expect > 1.2 * item["limit"]:
                mark(item_id, status="predicted_tl", predicted=round(expect))
                continue
            need = min(item["limit"], expect * 1.3 if expect else item.get("est", 240)) + 90
            if remaining < need:
                unfit.add(item_id)
                continue
            try:
                os.mkdir(os.path.join(Q, "lock", item_id))
            except FileExistsError:
                blocked += 1
                continue
            open(os.path.join(Q, "lock", item_id, "owner"), "w").write(f"{a.job} {a.hosts}\n")
            try:
                record = run_item(item, a.job, a.hosts)
            except Exception as e:             # launch problem: record it and go on
                record = dict(status="fail", message=repr(e)[:300])
            if record.get("status") != "ok" and left(a.job) < 90:
                # cut off by the end of the allocation, not a failure of its own: give the run back
                for f in glob.glob(os.path.join(Q, "lock", item_id, "*")):
                    os.remove(f)
                os.rmdir(os.path.join(Q, "lock", item_id))
                return
            mark(item_id, job=a.job, hosts=a.hosts, **record)
            ran = True
            break                              # start again from the top of the list
        if ran:
            idle = 0
            continue
        if blocked == 0:
            break
        idle += 1
        if idle % 6 == 0:
            clean_locks()
        time.sleep(10)


if __name__ == "__main__":
    main()
