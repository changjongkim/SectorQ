#!/usr/bin/env python3
"""Runner of the review runs on 128 GPUs (queue_z): jobs/g_run.py with its queue moved to queue_z and its logs to
logs/s, so that the figures read these runs with the other runs on 32 GPUs and more.

  python3 jobs/z_run.py --job <allocation id> --hosts <host,...>
  python3 jobs/z_run.py --count
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_run                                    # noqa: E402

g_run.Q = os.path.join(g_run.ROOT, "queue_z")
g_run.LOGS = os.path.join(g_run.ROOT, "logs", "s")
_plain = g_run.run_engine


def run_engine(item, job, hosts):
    record = _plain(item, job, hosts)
    record["log"] = f"logs/s/{item['id']}.out"
    return record


g_run.run_engine = run_engine
if __name__ == "__main__":
    g_run.main()
