from __future__ import annotations

def summarize_round(round_no: int, nodes: list[dict], policy_id: str) -> dict:
    scored = [n["composite"] for n in nodes if n.get("composite") is not None]
    spend = sum(float(n.get("cost", {}).get("gpu_hours", 0.0)) for n in nodes)
    return {"round": round_no, "policy_id": policy_id, "n": len(nodes),
            "best": max(scored) if scored else None, "spend": spend}

def print_summary(summary: dict) -> None:
    for s in summary["summaries"]:
        print(f"round {s['round']}: n={s['n']} best={s['best']} "
              f"spend={s['spend']:.3f} policy={s['policy_id']}")
