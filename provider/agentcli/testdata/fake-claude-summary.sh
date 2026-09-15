#!/bin/sh
# The real CLI puts a cumulative usage on the result line beside the per-turn
# figures. Two turns of 100 input and 10 output each, then the summary saying
# so, which must not be counted as a third turn.
printf '%s\n' '{"type":"system","subtype":"init","session_id":"s-1"}'
printf '%s\n' '{"type":"stream_event","session_id":"s-1","event":{"type":"message_start","message":{"id":"m1","model":"claude-test","usage":{"input_tokens":100}}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"one"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_start","message":{"id":"m2","model":"claude-test","usage":{"input_tokens":100}}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"two"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}}'
printf '%s\n' '{"type":"result","subtype":"success","session_id":"s-1","total_cost_usd":0.02,"num_turns":2,"duration_ms":1000,"usage":{"input_tokens":200,"output_tokens":20}}'
