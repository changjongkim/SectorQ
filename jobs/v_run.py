#!/usr/bin/env python3
"""Runner of the second review round (queue_v, logs/v): jobs/g_run.py with its queue and log directories moved, and
the paulistrings-rs runs (kind "prs") through jobs/p_drive.py's launch with its NCCL choice (queue_p/nccl).

  python3 jobs/v_run.py --job <allocation id> --hosts <host[,host...]>
  python3 jobs/v_run.py --count
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_run                                    # noqa: E402
import p_drive                                  # noqa: E402

g_run.Q = os.path.join(g_run.ROOT, "queue_v")
g_run.LOGS = os.path.join(g_run.ROOT, "logs", "v")
p_drive.LOGS = g_run.LOGS
for k in (2, 4, 8):
    p_drive.CIRCUITS[f"rpk{k}"] = (f"rpauli_k{k}", (), ())
_plain_engine, _plain_item = g_run.run_engine, g_run.run_item


def run_engine(item, job, hosts):
    record = _plain_engine(item, job, hosts)
    record["log"] = f"logs/v/{item['id']}.out"     # g_run writes the path of campaign g into the record
    return record


def run_item(item, job, hosts):
    if item["kind"] != "prs":
        return _plain_item(item, job, hosts)
    nccl = open(os.path.join(g_run.ROOT, "queue_p", "nccl")).read().strip()
    rec = p_drive.launch(item["id"], item["mode"], item["circuit"], item["cut"], job, hosts.split(",")[0], nccl)
    rec["log"] = f"logs/v/{item['id']}.out"
    rec.pop("job", None)                        # g_run.mark adds the allocation itself
    rec.pop("hosts", None)
    return rec


g_run.run_engine = run_engine
g_run.run_item = run_item
_supported = g_run.supported
g_run.supported = lambda item: item["kind"] == "prs" or _supported(item)

if __name__ == "__main__":
    os.makedirs(g_run.LOGS, exist_ok=True)
    g_run.main()
