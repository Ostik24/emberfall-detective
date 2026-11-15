#!/usr/bin/env bash
# PostToolUse hook: after Edit/Write on pipeline code or prompts, remind the agent
# that the change is not done until evals are re-run (AGENTS.md section 3).
input=$(cat)
path=$(printf '%s' "$input" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$path" in
  *concierge/*|*prompts/*|*mcp_server*)
    echo "Pipeline file changed: $path. Per AGENTS.md section 3, run the relevant evals on dev (run-evals skill) and log the CRISP-DM loop before calling this change done." >&2
    exit 2 ;;
esac
exit 0
