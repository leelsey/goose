#!/bin/sh
# A turn streamed as deltas, then restated whole on one line once it has
# finished. The binary itself restates a block at a time, and earlier, which is
# fake-claude-blocks.sh; a whole restatement must not repeat the turn either.
cat >/dev/null
printf '%s\n' '{"type":"system","subtype":"init","session_id":"s1"}'
printf '%s\n' '{"type":"stream_event","session_id":"s1","event":{"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":{"input_tokens":11}}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"whole "}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}}'
printf '%s\n' '{"type":"assistant","session_id":"s1","message":{"id":"msg_1","model":"claude-x","content":[{"type":"text","text":"whole answer"}]}}'
printf '%s\n' '{"type":"result","session_id":"s1","subtype":"success","result":"whole answer","total_cost_usd":0.002,"duration_ms":120}'
