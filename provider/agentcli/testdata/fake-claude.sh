#!/bin/sh
# A stand-in for the claude CLI: enough of the stream-json shape to pin the
# parser, including the two-turn accounting the real one produces.
printf '%s\n' '{"type":"system","subtype":"init","session_id":"sess-42"}'
printf '%s\n' '{"type":"stream_event","session_id":"sess-42","event":{"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":100}}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"considering"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"search"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":40}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_2","model":"claude-test","usage":{"input_tokens":200}}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"the answer"}}}'
printf '%s\n' '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}}'
printf '%s\n' '{"type":"result","subtype":"success","session_id":"sess-42","total_cost_usd":0.0123,"num_turns":2,"duration_ms":2500}'
