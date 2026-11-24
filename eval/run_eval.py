"""Evaluate retrieval and guardrails on the golden set; optionally the full agent with a live LLM.

Usage:
  python eval/run_eval.py              # retrieval + guardrails (no API key needed); used as the CI gate
  python eval/run_eval.py --with-llm   # also run the full agent: citation accuracy, review rate, latency
"""
import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from coverwise.guardrails import classify_intent
from coverwise.retrieval import get_retriever

HERE = Path(__file__).parent
GATES = {"hybrid_recall@5": 0.90, "intent_accuracy": 0.90}


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def retrieval_metrics(rows, k=5):
    retriever = get_retriever()
    out = {}
    for mode in ("bm25", "vector", "hybrid"):
        hits_at_1 = hits_at_k = rr = 0.0
        for row in rows:
            docs = [doc_of(h.chunk_id) for h in retriever.search(row["question"], k=k, mode=mode)]
            hits_at_1 += docs[:1] == [row["expected_doc"]]
            hits_at_k += row["expected_doc"] in docs
            rr += 1 / (docs.index(row["expected_doc"]) + 1) if row["expected_doc"] in docs else 0
        n = len(rows)
        out[mode] = {"recall@1": round(hits_at_1 / n, 3), f"recall@{k}": round(hits_at_k / n, 3),
                     "mrr": round(rr / n, 3)}
    return out


def _invoke_with_backoff(agent, payload, pause_s, attempts=6):
    """Free-tier APIs rate-limit (429/503). Wait and retry; latency counts only the successful attempt."""
    from coverwise.llm import LLMUnavailableError

    for attempt in range(attempts):
        try:
            t = time.perf_counter()
            state = agent.invoke(payload)
            return state, time.perf_counter() - t
        except LLMUnavailableError as err:
            if attempt == attempts - 1 or not any(code in str(err) for code in ("429", "503")):
                raise
            wait = 20 * (attempt + 1)
            print(f"  rate limited ({err}); waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
        finally:
            time.sleep(pause_s)
    raise AssertionError("unreachable")


def agent_metrics(coverage_rows, pause_s=0.0):
    from coverwise.api import get_agent

    agent = get_agent()
    latencies, cited, correct_doc, review = [], 0, 0, 0
    for i, row in enumerate(coverage_rows):
        state, seconds = _invoke_with_backoff(agent, {"request_id": f"eval-{i}", "question": row["question"]}, pause_s)
        latencies.append(seconds)
        print(f"  [{i + 1}/{len(coverage_rows)}] {seconds:.1f}s citations={state['citations']}", file=sys.stderr)
        cited += bool(state["citations"])
        correct_doc += row["expected_doc"] in {doc_of(c) for c in state["citations"]}
        review += state["needs_review"]
    n = len(coverage_rows)
    q = statistics.quantiles(latencies, n=20)
    return {"citation_rate": round(cited / n, 3), "cited_expected_doc": round(correct_doc / n, 3),
            "needs_review_rate": round(review / n, 3), "latency_p50_s": round(statistics.median(latencies), 2),
            "latency_p95_s": round(q[18], 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between LLM questions (free-tier limits)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line]
    coverage = [r for r in rows if r["intent"] == "coverage"]
    intent_ok = sum(classify_intent(r["question"]) == r["intent"] for r in rows)

    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "golden_questions": len(rows),
        "retrieval": retrieval_metrics(coverage),
        "guardrails": {"intent_accuracy": round(intent_ok / len(rows), 3)},
    }
    if args.with_llm:
        results["agent"] = agent_metrics(coverage, args.pause)

    checks = {"hybrid_recall@5": results["retrieval"]["hybrid"]["recall@5"],
              "intent_accuracy": results["guardrails"]["intent_accuracy"]}
    results["gates"] = {name: {"value": v, "min": GATES[name], "pass": v >= GATES[name]} for name, v in checks.items()}

    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    failed = [n for n, g in results["gates"].items() if not g["pass"]]
    if failed:
        print(f"EVAL GATE FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
