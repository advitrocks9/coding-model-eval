import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/mellum_sft_multiturn.jsonl")
rows = [json.loads(l) for l in path.open()]
retried = [r for r in rows if not r.get("skipped_reason")]
print(f"retried: {len(retried)}")

for tag, filt in [("recovered", lambda r: r.get("recovered_at")),
                  ("never recovered", lambda r: not r["final_plus_passed"])]:
    print(f"\n===== examples: {tag} =====")
    for r in [x for x in retried if filt(x)][:3]:
        print(f"\n--- {r['task_id']} (recovered_at={r.get('recovered_at')}) ---")
        for t in r["turns"]:
            print(f"  turn {t['turn']} base={t['base_passed']} plus={t['plus_passed']} kind={t['plus_kind']}")
            short = t["completion"].strip().splitlines()[:3]
            print(f"    > {' / '.join(short)[:160]}")
