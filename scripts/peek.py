"""Quick stats on a multi-turn JSONL during a run."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/mellum_sft_multiturn.jsonl")
rows = [json.loads(l) for l in path.open() if l.strip()]
n = len(rows)
retried = [r for r in rows if not r.get("skipped_reason")]
recov = [r for r in retried if r.get("recovered_at") is not None]
print(f"rows: {n}")
print(f"retried (non-skip): {len(retried)}")
print(f"recovered: {len(recov)}")
print(f"final_pass: {sum(r['final_plus_passed'] for r in rows)}")
if recov:
    from collections import Counter
    ts = Counter(r["recovered_at"] for r in recov)
    print(f"recovery turn dist: {dict(ts)}")
