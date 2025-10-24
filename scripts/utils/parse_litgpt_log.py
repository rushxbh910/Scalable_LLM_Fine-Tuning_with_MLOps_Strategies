#!/usr/bin/env python3
import sys, json, re, os
log_path, summary_path, key = sys.argv[1], sys.argv[2], sys.argv[3]
with open(log_path, "r", errors="ignore") as f:
    text = f.read()

# Heuristics: grab last 'Total time' / 'time' like entries and max memory
time_match = re.findall(r"(Total|total|Training) time[:=]\s*([0-9]+\.[0-9]+|\d+)\s*(s|sec|seconds|)", text)
mem_match  = re.findall(r"(Max|max).{0,20}(memory|RAM|mem).{0,20}([0-9]+(\.[0-9]+)?)\s*(GB|GiB)", text)

def pick(m):
    return m[-1][1] if m else None

def pick_mem(m):
    return m[-1][2] if m else None

summary = {}
if os.path.exists(summary_path):
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except:
        summary = {}

summary[key] = {
    "training_time_sec": float(pick(time_match)) if time_match else None,
    "max_mem_gb": float(pick_mem(mem_match)) if mem_match else None,
    "log": os.path.basename(log_path),
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {summary_path}")
