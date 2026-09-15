#!/bin/sh
# One invocation the binary billed as five turns while streaming frames goose
# cannot tell apart: the tool calls it made in between never reached the wire.
# The result line is the only place the real count appears.
printf '%s\n' '{"type":"system","subtype":"init","session_id":"s-1"}'
printf '%s\n' '{"type":"stream_event","session_id":"s-1","event":{"type":"message_start","message":{"id":"m1","model":"claude-test","usage":{"input_tokens":100}}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}}'
printf '%s\n' '{"type":"result","subtype":"success","session_id":"s-1","total_cost_usd":0.05,"num_turns":5,"duration_ms":1000,"usage":{"input_tokens":100,"output_tokens":10}}'
