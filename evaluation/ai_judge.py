"""
ai_judge.py
Reusable helpers for LLM-as-Judge evaluation that work **with or without** reference answers.

Example (API)
--------------
>>> from ai_judge import AIJudge
>>> judge = AIJudge()
>>> judge.evaluate("Paris is in Germany.", "Paris is in France.")
{'accuracy': 0.0, 'coherence': 0.92, 'clarity': 0.83}
>>> judge.evaluate("This is my standalone summary.")
{'clarity': 0.86}

Example (CLI)
-------------
$ python -m ai_judge --prediction "Paris is in Germany." --reference "Paris is in France."
accuracy: 0.00
coherence: 0.92
clarity: 0.83
weighted (equal-weights): 0.58
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# metric_name -> (rubric_prompt, requires_reference)
DEFAULT_RUBRICS: Mapping[str, tuple[str, bool]] = {
    "accuracy": (
        "Evaluate factual correctness between the prediction and the reference. "
        "Return only a number from 0 (totally wrong) to 1 (completely correct).",
        True,
    ),
    "coherence": (
        "Evaluate logical consistency between the prediction and the reference. "
        "Return only a number 0-1.",
        True,
    ),
    "clarity": (
        "Rate how clear and easy to understand the prediction is (no reference needed). "
        "Return only a number 0-1.",
        False,
    ),
}


class AIJudge:
    """LLM-powered evaluator using configurable rubric metrics.

    Parameters
    ----------
    llm : ChatOpenAI | None
        LangChain chat model; defaults to GPT-4o temperature-0.
    rubrics : Mapping[str, tuple[str,bool]] | None
        Mapping metric → (rubric_prompt, requires_reference).
    """

    def __init__(self, *, llm: Optional[ChatOpenAI] = None,
                 rubrics: Optional[Mapping[str, tuple[str, bool]]] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.rubrics: Dict[str, tuple[str, bool]] = dict(rubrics or DEFAULT_RUBRICS)

    def evaluate(self, prediction: str, reference: Optional[str] = None,
                 *, include_prompts: bool = False) -> Dict[str, float]:
        """Score *prediction* (optionally against *reference*).

        Returns a dict {metric_name: score}. Metrics requiring a reference are
        skipped if *reference* is None or empty.
        """
        scores: Dict[str, float] = {}
        if include_prompts:
            scores["_prompts"] = {}
        for name, (rubric, needs_ref) in self.rubrics.items():
            if needs_ref and not reference:
                continue
            prompt = self._prompt(rubric, prediction, reference)
            if include_prompts:
                scores["_prompts"][name] = "\n".join(m.content for m in prompt)
            raw = self.llm.invoke(prompt).content.strip()
            try:
                val = float(raw)
            except ValueError:
                val = 0.0
            scores[name] = max(0.0, min(1.0, val))
        return scores

    @staticmethod
    def _prompt(rubric: str, prediction: str, reference: Optional[str]) -> List[SystemMessage | HumanMessage]:
        ref_block = f"Reference answer:\n{reference}\n\n" if reference else ""
        return [
            SystemMessage(content="You are an impartial evaluator. " + rubric),
            HumanMessage(content=f"{ref_block}Model prediction:\n{prediction}\n\nScore:")
        ]

    def add_metric(self, name: str, rubric_prompt: str, requires_reference: bool = True):
        """Add or override a rubric metric at runtime."""
        self.rubrics[name] = (rubric_prompt, requires_reference)

    def available_metrics(self) -> List[str]:
        return list(self.rubrics.keys())

    @staticmethod
    def _cli_parse() -> argparse.Namespace:
        p = argparse.ArgumentParser(description="AI-Judge – rubric-based evaluation")
        p.add_argument("--prediction", required=True, help="Model prediction text or @file.txt")
        p.add_argument("--reference", help="Reference text or @file.txt (omit for no-reference)")
        p.add_argument("--weights", nargs="*", default=[],
                       help="metric=weight pairs, e.g. accuracy=0.5 clarity=0.5")
        return p.parse_args()

    @staticmethod
    def _read_arg(val: str | None) -> Optional[str]:
        if not val:
            return None
        if val.startswith("@"):
            return Path(val[1:]).read_text()
        return val

    def _weighted(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        if not weights:
            weights = {m: 1.0 for m in scores}
        active = {m: w for m, w in weights.items() if m in scores}
        total = sum(active.values()) or 1.0
        return sum(scores[m] * active[m] for m in active) / total

    def _parse_weights(self, weight_args: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for pair in weight_args:
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            try:
                out[k.strip().lower()] = float(v)
            except ValueError:
                continue
        return out

    def _main(self):
        args = self._cli_parse()
        pred = self._read_arg(args.prediction)
        ref = self._read_arg(args.reference)
        if pred is None:
            print("Prediction is required", file=sys.stderr)
            sys.exit(1)
        scores = self.evaluate(prediction=pred, reference=ref)
        weights = self._parse_weights(args.weights)
        weighted = self._weighted(scores, weights)
        for k, v in scores.items():
            print(f"{k}: {v:.2f}")
        if len(scores) > 1:
            print(f"weighted ({'equal-weights' if not weights else 'custom'}): {weighted:.2f}")

if __name__ == "__main__":
    AIJudge()._main()
