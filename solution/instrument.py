"""Helpers to wire the Day 13 telemetry toolkit around the opaque agent.
Import these from wrapper.py. The agent emits NOTHING, so whatever you record here
is the only telemetry you will have to diagnose the faults.
"""
from __future__ import annotations
import time
import re

try:
    from telemetry.logger import logger, new_correlation_id, set_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
except Exception:  # telemetry is optional; wrapper still runs without it
    logger = None

    def cost_from_usage(model, usage):
        return 0.0

    def redact(s):
        return (s, 0)


def _analyze_answer(answer: str) -> dict:
    """Deeper analysis of answer quality for diagnostics."""
    if not answer:
        return {"has_total": False, "total_pattern": None, "is_refusal": False}
    
    answer_lower = answer.lower()
    # Check for correct total format: "Tong cong: <int> VND"
    total_match = re.search(r'tong\s+cong\s*:\s*(\d+)\s*vnd', answer_lower)
    refusal_keywords = ['không thể', 'không hoàn tất', 'hết hàng', 'không có hàng', 'không hỗ trợ']
    is_refusal = any(kw in answer_lower for kw in refusal_keywords)
    
    return {
        "has_total": bool(total_match),
        "total_value": int(total_match.group(1)) if total_match else None,
        "is_refusal": is_refusal,
        "answer_length": len(answer),
    }


def observed_call(call_next, question, config, context):
    """Enhanced logging: time the call, compute cost, check PII, analyze answer quality."""
    t0 = time.time()
    res = call_next(question, config)
    wall_ms = int((time.time() - t0) * 1000)
    meta = res.get("meta", {})
    usage = meta.get("usage", {})
    answer = res.get("answer") or ""
    answer_analysis = _analyze_answer(answer)
    
    if logger:
        logger.log_event("AGENT_CALL", {
            "qid": context.get("qid"),
            "status": res.get("status"),
            "reported_latency_ms": meta.get("latency_ms"),
            "wall_ms": wall_ms,
            "tokens": usage,
            "cost_usd": cost_from_usage(meta.get("model", ""), usage),
            "pii_in_answer": redact(answer)[1] > 0,
            "tools_used": meta.get("tools_used", []),
            "has_valid_total": answer_analysis["has_total"],
            "is_refusal": answer_analysis["is_refusal"],
            "answer_length": answer_analysis["answer_length"],
            "steps": res.get("steps", 0),
            "error": meta.get("error"),
        })
    return res
