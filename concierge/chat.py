"""Run the Movie Detective locally, in the terminal or in a browser, with a live trace.

    uv run python -m concierge.chat                       # terminal chat with the trace
    uv run --with gradio python -m concierge.chat --web   # browser UI on http://127.0.0.1:7860

Purpose: the demo for the video. Every turn prints what happened inside: each tool the agent
called (with its arguments), the picks with the verifier's verdicts and the explanation, and
the turn's API calls, cost and latency. Questions answered before replay from the disk cache
(free, instant); new ones need EMBERFALL_CLAUDE_KEY and TMDB_API_KEY in `.env`.
Input: user messages (EN/UK). Output: the agent's reply plus the trace.
"""

import argparse
import json

from concierge import config
from concierge.agent import AgentTurn, DetectiveAgent
from concierge.finalize import verify_and_explain
from concierge.tools.registry import Toolbox


def model_calls(log_lines: list[str]) -> str:
    """Model calls of one turn from the usage log: model, prompt (= component), cached or not."""
    rows = [json.loads(line) for line in log_lines]
    seen: dict[tuple[str, str, bool], int] = {}
    for r in rows:
        key = (r["model"], r["prompt_version"], r["from_cache"])
        seen[key] = seen.get(key, 0) + 1
    return "\n".join(
        f"  model: {model} {prompt} x{n}{' (cache)' if cached else ''}"
        for (model, prompt, cached), n in seen.items()
    )


def trace(turn: AgentTurn) -> str:
    """The inside of one turn, as plain text."""
    lines = []
    for n, call in enumerate(turn.tool_calls, start=1):
        args = json.dumps(call.input, ensure_ascii=False)
        status = "ok" if call.ok else f"ERROR {call.error}"
        lines.append(f"  tool {n}: {call.name}({args[:160]}) -> {status}")
    for pick in turn.picks:
        lines.append(f"  pick: {pick.id} {pick.title} ({pick.year})")
        verification = pick.notes.get("verification") or {}
        verdicts = [c["verdict"] for c in verification.get("checks", [])]
        if verdicts:
            counts = {v: verdicts.count(v) for v in dict.fromkeys(verdicts)}
            lines.append(
                f"    verifier: {counts}; false memory: {verification.get('false_memory')}"
            )
        if explanation := pick.notes.get("explanation"):
            lines.append(f"    explanation: {explanation}")
    if turn.dropped_picks:
        lines.append(f"  blocked picks (no tool returned them): {turn.dropped_picks}")
    lines.append(
        f"  {turn.api_calls} API calls (rest from cache), ${turn.cost_usd:.4f}, "
        f"{turn.latency_s:.1f} s{', respond forced' if turn.forced_respond else ''}"
    )
    return "\n".join(lines)


def reply_text(turn: AgentTurn) -> str:
    text = turn.message
    if turn.options:
        text += "\n\nOptions: " + " / ".join(turn.options)
    return text


def run_turn(agent: DetectiveAgent, message: str) -> str:
    """The reply and the full trace (tools, picks, model calls) of one turn."""
    before = _log_size()
    turn = agent.turn(message)
    calls = model_calls(_log_lines_since(before))
    return f"{reply_text(turn)}\n\n--- trace ---\n{trace(turn)}\n{calls}"


def _log_size() -> int:
    log = config.LLM_USAGE_LOG
    return log.stat().st_size if log.exists() else 0


def _log_lines_since(offset: int) -> list[str]:
    with config.LLM_USAGE_LOG.open(encoding="utf-8") as f:
        f.seek(offset)
        return [line for line in f.read().splitlines() if line.strip()]


def terminal(agent: DetectiveAgent) -> None:
    print("Emberfall Movie Detective. Describe a film you half remember (EN/UK). Empty line quits.")
    while message := input("\nyou> ").strip():
        print(f"\ndetective> {run_turn(agent, message)}")


def web(agent: DetectiveAgent) -> None:
    import gradio as gr

    def chat(message: str, history) -> str:
        reply, _, inside = run_turn(agent, message).partition("\n\n--- trace ---\n")
        return f"{reply}\n\n```\n{inside}\n```"

    gr.ChatInterface(chat, title="Emberfall Movie Detective").launch(show_error=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--web", action="store_true", help="browser UI (needs gradio)")
    args = parser.parse_args()
    for name in (config.CLAUDE_KEY_NAME, "TMDB_API_KEY"):
        try:
            config.get_secret(name)  # read once up front; the web UI answers in worker threads
        except config.MissingSecretError:
            print(f"note: {name} not set; only cached questions will work")
    agent = DetectiveAgent(Toolbox.create(), finalize=verify_and_explain)
    if args.web:
        web(agent)
    else:
        terminal(agent)


if __name__ == "__main__":
    main()
