#!/bin/sh
# The documented stream-json shape without partial-message wrappers: the whole
# assistant turn on one line, then the line that closes the invocation.
cat >/dev/null
printf '%s\n' '{"type":"assistant","session_id":"s1","message":{"id":"msg_1","model":"claude-x","content":[{"type":"text","text":"whole answer"}],"usage":{"input_tokens":11,"output_tokens":4}}}'
printf '%s\n' '{"type":"result","session_id":"s1","subtype":"success","result":"whole answer","total_cost_usd":0.002,"duration_ms":120,"usage":{"input_tokens":11,"output_tokens":4}}'
