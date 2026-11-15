"""Build the Ukrainian known-item set (`known_item_uk`) for the EN/UK gap.

    uv run python data/build_known_item_uk.py

Source: 50 dev + 50 test known_item queries with the answer in the catalog (fixed hash order).
Sonnet translates each into the Ukrainian a person would type: informal, same hedges and
mistakes, names kept as they would write them. Labels: `llm_draft` until the owner checks a
sample (then `llm_draft_checked`, brief section 8.4). Each translation keeps the split of its
English original, so the EN/UK comparison is paired.
Output: evals/gold/known_item_uk.jsonl.
"""

import hashlib

from concierge.config import SONNET
from concierge.llm import LLM
from evals.gold import KnownItem, load, write

PROMPT_VERSION = "translate-uk-v1"
PER_SPLIT = 50

SYSTEM = """Translate a person's message from English into Ukrainian, as a Ukrainian speaker \
would type it to a chatbot: natural and informal, keeping the same details, hedges, typos in \
spirit and mistakes (do not fix wrong facts). Keep film and actor names the way a Ukrainian \
would write them (usually the common Ukrainian spelling, or Latin if unsure). Output only the \
translation."""


def main() -> None:
    llm = LLM()
    examples: list[KnownItem] = []
    for split in ("dev", "test"):
        items = [e for e in load("known_item", split=split) if isinstance(e, KnownItem)]
        items = sorted((e for e in items if e.in_catalog),
                       key=lambda e: hashlib.sha256(e.id.encode()).hexdigest())  # fmt: skip
        for example in items[:PER_SPLIT]:
            result = llm.complete(SONNET, PROMPT_VERSION, SYSTEM,
                                  [{"role": "user", "content": example.query}],
                                  max_tokens=4000, effort="low")  # fmt: skip
            examples.append(
                example.model_copy(
                    update={
                        "id": f"uk-{example.id}",
                        "lang": "uk",
                        "source": f"llm_draft:{example.source.removeprefix('dataset:')}",
                        "query": result.text.strip(),
                        "notes": "Sonnet translation of the English original",
                    }
                )
            )
    path = write("known_item_uk", examples)
    print(f"wrote {path}: {len(examples)} examples; {llm.spend}")


if __name__ == "__main__":
    main()
