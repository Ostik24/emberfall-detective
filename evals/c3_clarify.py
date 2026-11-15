"""C3 clarifier: does asking one or two questions put the right title first?

Protocol (brief 7.2, CRS user-simulator literature): for each dev known-item query with the
answer in the catalog, take the top 6 of the deployed search; ask up to two questions; an
LLM user simulator that knows the hidden target answers each by picking one offered option
or "not sure" (options are an enum, so it cannot name the title); candidates are narrowed to
those the chosen option keeps.
Per-case metrics:
    s1_q0 / s1_q1 / s1_q2   answer at rank 1 after 0 / 1 / 2 questions
    answer_kept             (answer in the initial top 6) still among the candidates at the end
    redundant               the simulator judged a question already answered by the recollection
Simulator: Haiku with the target's catalog text + TMDB facts; prompt `simulator-v1`.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from concierge.clarify import NOT_SURE, PROMPT_VERSION, clarify, narrow
from concierge.config import CLARIFIER_MODEL
from concierge.finalize import evidence_text
from concierge.llm import LLM
from concierge.tools.registry import Toolbox
from concierge.tools.search_catalog import SearchCatalogArgs, key_str
from evals.gold import KnownItem, Split, load
from evals.metrics.stats import paired_difference

METRICS = ("s1_q0", "s1_q1", "s1_q2", "answer_kept", "redundant")
GROUPS = ("source", "in_top6")
TOP = 6
MAX_QUESTIONS = 2
SIM_PROMPT = "simulator-v1"

SIM_SYSTEM = """You play a person who half remembers a film or TV show and is looking for its \
title. The facts below describe the title you saw. Someone asks you a question with answer \
options.

Pick the option that fits the title you saw. If none fits, or the facts do not tell, answer \
"not sure" - real people are often unsure. Never reveal the title.
Also say whether your original description (below) already answered this question."""


class SimAnswer(BaseModel):
    answer: str
    already_said: bool


def simulate(llm: LLM, recollection: str, target_facts: str, question, options) -> SimAnswer:
    """The simulated user's answer: one of `options` or "not sure"."""
    tool = {
        "name": "answer",
        "description": "Your answer and whether your description already answered it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": [*options, NOT_SURE]},
                "already_said": {"type": "boolean"},
            },
            "required": ["answer", "already_said"],
        },
    }
    user = (
        f"<title_you_saw>\n{target_facts}\n</title_you_saw>\n\n"
        f"<your_description>\n{recollection}\n</your_description>\n\n"
        f"Question: {question}\nOptions: {'; '.join(options)}"
    )
    messages = [{"role": "user", "content": user}]
    return llm.call_tool(CLARIFIER_MODEL, SIM_PROMPT, SIM_SYSTEM, messages, tool, SimAnswer)


@dataclass(frozen=True)
class ClarifyConfig:
    parent: str | None = None


CONFIGS = {"default": ClarifyConfig()}


def run(config_name: str, split: Split) -> dict:
    box = Toolbox.create()
    llm, search = box.llm, box.search
    cases = []
    for example in load("known_item", split=split):
        assert isinstance(example, KnownItem)
        if not example.in_catalog:
            continue
        answer = f"{example.answer_item_type}/{example.answer_item_id}"
        rows, _ = search.rank(SearchCatalogArgs(query=example.query))
        top = [key_str(box.catalog.keys[r]) for r in rows[:TOP]]
        texts = [search.texts[r] for r in rows[:TOP]]
        current = list(range(len(top)))
        s1 = [float(top[0] == answer)]
        asked, redundant, dialog = [], False, []
        for _ in range(MAX_QUESTIONS):
            if len(current) < 2:
                s1.append(s1[-1])
                continue
            question = clarify(llm, example.query, [texts[i] for i in current], asked)
            sim = simulate(llm, example.query, evidence_text(box, answer), question.question,
                           question.option_texts())  # fmt: skip
            local = narrow(question, sim.answer, list(range(len(current))))
            current = [current[i] for i in local]
            asked.append(question.question)
            redundant |= sim.already_said
            dialog.append({"q": question.question, "a": sim.answer, "left": len(current)})
            s1.append(float(top[current[0]] == answer))
        in_top = answer in top
        cases.append(
            {
                "id": example.id,
                "source": example.source,
                "in_top6": str(in_top),
                "dialog": dialog,
                "s1_q0": s1[0],
                "s1_q1": s1[1],
                "s1_q2": s1[2],
                "answer_kept": float(answer in [top[i] for i in current]) if in_top else None,
                "redundant": float(redundant) if dialog else None,
            }
        )
    return {
        "config": {
            "name": config_name,
            "prompt": PROMPT_VERSION,
            "simulator": SIM_PROMPT,
            "search": search.config.__dict__,
        },
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["answer_kept"] == 0.0],
        "llm": llm.spend.model_dump(),
    }


def summarize(cases: list[dict]) -> dict:
    """Paired gain of asking, with a bootstrap CI over queries."""
    gains = {}
    for after in ("s1_q1", "s1_q2"):
        mean, low, high = paired_difference([c["s1_q0"] for c in cases], [c[after] for c in cases])
        gains[f"{after}-s1_q0"] = [round(mean, 4), round(low, 4), round(high, 4)]
    return {"gain_with_ci": gains}
