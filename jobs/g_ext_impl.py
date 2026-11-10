#!/usr/bin/env python3
"""Campaign g: one run of an external baseline.   g_ext_impl.py --job J --hosts h[,h..] --item '<json>'

Fields of the item: backend (spd, orqa, monoprop, julia, ppvm, qiskit, obp), case (circuit), coef (cutoff), policy,
rpn (ranks per node), threads (per rank), limit (s); optional: use (nodes the program is started on; the other
hosts given are left to it but stay idle), gpus (GPUs visible to SPD), prof (time the MPI calls with
bin/libmpi_shim.so), sample (period in s of the CPU counters, and of nvidia-smi for SPD, on every host given),
gates (per-gate times of the adapter).

Adapters, pinned sources and environments are those of the worktree kcj/pauliprop-sectorq-impl (read only); every
implementation receives the operation list of SectorQ. Output: results/g_ext/<id>.{launch.json,json,ir.json,stdout.txt,
stderr.txt,mpiprof} and logs/g/{cpu,smi}.<id>.<host>. The last line printed is the record for the queue.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPL = "/pscratch/sd/s/sgkim/kcj/pauliprop-sectorq-impl"
OUT = os.path.join(ROOT, "results", "g_ext")
LOGS = os.path.join(ROOT, "logs", "g")


def solved(result):
    """The adapter's result file exists and reports a completed solve."""
    try:
        done = json.load(open(result))
    except (OSError, ValueError):
        return False
    return done.get("status") == "success" and bool(done.get("results"))


def stop(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def launch_engine(item, job, hosts):
    """Several builds or settings of the engine, one after the other on the same hosts (backend "engine").
    variants: [name, binary in bin/, extra arguments]; logs/<tag>/<name>.<id>.out in the format of logs/g."""
    nn = len(hosts.split(","))
    tag = item.get("tag", "h1")
    os.makedirs(os.path.join(ROOT, "logs", tag), exist_ok=True)
    env = os.environ.copy()
    env["MPICH_GPU_SUPPORT_ENABLED"] = "0"
    walls, values, codes = {}, {}, {}
    start = time.time()
    for name, binary, extra in item["variants"]:
        log = os.path.join(ROOT, "logs", tag, f"{name}.{item['id']}.out")
        cmd = ["timeout", "300", "srun", f"--jobid={job}", f"--nodelist={hosts}", "--overlap", "-N", str(nn), "-n", str(nn),
               "--ntasks-per-node=1", "-c", "128", "--gpus-per-node=4", "--cpu-bind=none"]
        if item.get("prof"):                   # time the MPI calls of the engine (bin/libmpi_shim.so)
            cmd.append(f"--export=ALL,LD_PRELOAD={ROOT}/bin/libmpi_shim.so,PP_MPIPROF_OUT={ROOT}/logs/{tag}/{name}.{item['id']}.mpiprof")
        cmd += ["stdbuf", "-oL", "./bin/" + binary, "--gpus", str(item["per"])] + (item["args"] + " " + extra).split()
        with open(log, "w") as f:
            f.write(f"===== engine nodes={nn} gpus={nn * item['per']} args: {item['args']} {extra} --gpus {item['per']} binary={binary}\n")
            f.flush()
            code = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            f.write(f"exit={code}\n")
        text = open(log, errors="replace").read()
        w = re.search(r"^wall=([0-9.]+)", text, re.M)
        v = re.search(r"^expectation=(\S+)", text, re.M)
        walls[name], values[name], codes[name] = (float(w.group(1)) if w else None), (float(v.group(1)) if v else None), code
    same = len({round(v, 10) for v in values.values() if v is not None}) == 1
    ok = all(c == 0 for c in codes.values()) and same and all(v is not None for v in values.values())
    return dict(status="ok" if ok else "fail", walls=walls, values=values, exits=codes, seconds=round(time.time() - start, 1), log=f"logs/{tag}/")


def launch(item, job, hosts):
    if item["backend"] == "engine":
        return launch_engine(item, job, hosts)
    given = hosts.split(",")
    use = int(item.get("use", len(given)))
    sel = given[:use]
    backend, case, policy = item["backend"], item["case"], item["policy"]
    rpn, threads, limit = int(item["rpn"]), int(item["threads"]), float(item["limit"])
    os.makedirs(OUT, exist_ok=True)
    name, n = item["id"], 0
    while os.path.exists(os.path.join(OUT, name + ".launch.json")):          # a run cut off by the end of an allocation
        n += 1
        name = f"{item['id']}.retry{n}"
    path = lambda ext: os.path.join(OUT, name + ext)
    source = os.path.join(ROOT, "results", "f_ext", "ir", case + ".ir.json")
    if not os.path.exists(source):
        source = os.path.join(IMPL, "results", "own_perf_v1_n1", case + "_engine_r0.ir.json")
    ir = json.load(open(source))
    ir["truncation"]["cutoff"] = float(item["coef"])
    open(path(".ir.json"), "w").write(json.dumps(ir) + "\n")
    step = ["srun", f"--jobid={job}", "--overlap"]
    prefix = step + ["-N", str(use), "-n", str(use * rpn), f"--ntasks-per-node={rpn}", "--nodelist=" + ",".join(sel),
                     "-c", str(2 * threads), "--cpu-bind=cores", "--kill-on-bad-exit=1"]
    if backend == "spd":
        prefix.append(f"--gpus-per-node={int(item.get('gpus', 1))}")
    if item.get("prof"):
        prefix.append(f"--export=ALL,LD_PRELOAD={ROOT}/bin/libmpi_shim.so,PP_MPIPROF_OUT={path('.mpiprof')}")
    env = os.environ.copy()
    env.update(MPICH_GPU_SUPPORT_ENABLED="0", monoprop_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS=str(threads))
    result = path(".json")
    common = ["--input", path(".ir.json"), "--output", result, "--policy", policy, "--repeats", "1"]
    if backend == "orqa":
        flat = path(".flat")
        subprocess.run(step + ["-N", "1", "-n", "1", "--nodelist=" + sel[0], "-c", "1", IMPL + "/env/sota/bin/python",
                               IMPL + "/bench/ir_flat.py", path(".ir.json"), flat], env=env, check=True, timeout=180)
        command = prefix + [IMPL + "/bin/orqa_reproduction", "--input", flat, "--policy", "matched" if policy == "matched" else "relative"]
    elif backend in ("spd", "monoprop"):
        command = prefix + [IMPL + "/env/sota/bin/python", IMPL + "/bench/sota_adapter.py", "--backend", backend] + common
    elif backend == "qiskit":
        command = prefix + [IMPL + "/env/auxproject/.venv/bin/python", IMPL + "/bench/aux_adapter.py", "--backend", backend] + common
    elif backend == "obp":
        command = prefix + [IMPL + "/env/obp/bin/python", IMPL + "/bench/obp_adapter.py", "--backend", backend] + common + \
            ["--max-seconds", str(max(1, int(limit) - 30))]
    elif backend == "ppvm":
        command = prefix + [IMPL + "/bin/ppvm_parallel"] + common
    elif backend == "julia":
        env["JULIA_DEPOT_PATH"] = IMPL + "/env/julia-depot"
        command = prefix + [IMPL + "/env/julia/bin/julia", "--project=" + IMPL + "/env/julia-project", f"--threads={threads}",
                            IMPL + "/bench/julia_adapter.jl"] + common
    else:
        raise ValueError(backend)
    if item.get("gates") and backend != "orqa":
        command.append("--profile-gates")
    record = {"schema": "sectorq.g_ext.v1", "name": name, "backend": backend, "case": case, "policy": policy, "cutoff": float(item["coef"]),
              "hosts_given": given, "hosts_used": sel, "ranks": use * rpn, "threads_per_rank": threads, "timeout_seconds": limit,
              "gpus_visible": int(item.get("gpus", 1)) if backend == "spd" else 0, "command": command, "job": job,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running"}
    json.dump(record, open(path(".launch.json"), "w"), indent=1)
    samplers = []
    if item.get("sample"):
        all_hosts = step + ["-N", str(len(given)), "-n", str(len(given)), "--ntasks-per-node=1", "--nodelist=" + ",".join(given), "-c", "1"]
        quiet = dict(cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
        samplers.append(subprocess.Popen(all_hosts + ["bash", "jobs/g_cpu.sh", os.path.join(LOGS, "cpu." + name), str(item["sample"])], **quiet))
        if backend == "spd":
            samplers.append(subprocess.Popen(all_hosts + ["--gpus-per-node=4", "bash", "jobs/g_smi.sh", os.path.join(LOGS, "smi." + name),
                                                          str(int(1000 * float(item["sample"])))], **quiet))
        time.sleep(2)
    start = time.perf_counter()
    with open(path(".stdout.txt"), "w") as so, open(path(".stderr.txt"), "w") as se:
        proc = subprocess.Popen(command, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL, start_new_session=True)
        # MonoProp can hang in MPI teardown after it wrote its result (queue_s/done_prev, job 61024060: two runs held
        # 16 nodes to the limit). Once the adapter's result file reports success, the run gets 60 s to exit and is then
        # stopped; the result below is read from that file.
        code, seen = None, None
        while code is None:
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                waited = time.perf_counter() - start
                if seen is None and backend != "orqa" and solved(result):
                    seen = waited
                if waited > limit or (seen is not None and waited - seen > 60):
                    break
        if code is not None:
            record.update(exit_code=code, status="success" if code == 0 else "failed")
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            record.update(exit_code=proc.returncode, status="timeout" if seen is None else "failed")
            if seen is not None:
                record["stopped_after_result"] = True
    record["cold_end_to_end_seconds"] = time.perf_counter() - start
    for s in samplers:
        stop(s)
    stdout, stderr = open(path(".stdout.txt")).read(), open(path(".stderr.txt")).read()
    if record["status"] == "failed" and re.search(r"out of memory|OutOfMemory|oom-kill|oom_kill|std::bad_alloc|Killed", stdout + stderr, re.I):
        record["status"] = "oom"
    if record["status"] == "failed" and "NotImplementedError" in stderr:
        record["status"] = "unsupported"
    if backend != "orqa" and os.path.exists(result) and record["status"] != "success":
        done = json.load(open(result))               # the adapter writes each completed solve
        if done.get("status") == "success" and done.get("results"):
            record["status"] = "success"
    if record["status"] == "success":
        if backend == "orqa":
            parsed = {k: float(v) for k, v in re.findall(r"(\w+)=([-+\d.eE]+)", stdout)}
            record.update(expectation=parsed.get("expectation"), solver_seconds=parsed.get("wall"), metrics=parsed)
        else:
            payload = json.load(open(result))
            record.update(expectation=payload["results"][0]["expectation"], solver_seconds=payload["first_solver_seconds"],
                          reported_threads=payload.get("threads"), first_result={k: v for k, v in payload["results"][0].items()
                                                                                 if not isinstance(v, (list, dict))})
            if payload.get("status") != "success":
                record["status"] = payload["status"]
    if os.path.exists(path(".mpiprof")):
        record["mpi"] = open(path(".mpiprof")).read().splitlines()
    json.dump(record, open(path(".launch.json"), "w"), indent=1)
    status = {"success": "ok", "timeout": "tl", "failed": "fail"}.get(record["status"], record["status"])
    return dict(status=status, exit=record.get("exit_code"), wall=record.get("solver_seconds"), expectation=record.get("expectation"),
                seconds=round(record["cold_end_to_end_seconds"], 1), log=f"results/g_ext/{name}.launch.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--hosts", required=True)
    ap.add_argument("--item", required=True)
    a = ap.parse_args()
    try:
        out = launch(json.loads(a.item), a.job, a.hosts)
    except Exception as e:
        out = dict(status="fail", message=repr(e)[:300])
    print(json.dumps(out), flush=True)
