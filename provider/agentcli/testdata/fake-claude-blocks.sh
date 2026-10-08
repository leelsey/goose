#!/bin/sh
# What the claude CLI prints with --include-partial-messages for a message that
# thinks, answers and calls a tool, then for the message that follows once the
# tool has run. The binary prints an "assistant" line as each block closes,
# holding that block alone under the message's id with the usage message_start
# announced, and all of them before the message_delta that finishes it.
# With GOOSE_TEST_WHOLE set the stream events are left out, which is what the
# binary prints without --include-partial-messages.
cat >/dev/null
ev() { [ -n "$GOOSE_TEST_WHOLE" ] || printf '%s\n' "$1"; }
line() { printf '%s\n' "$1"; }
U1='{"input_tokens":10,"cache_read_input_tokens":1000,"output_tokens":1}'
U2='{"input_tokens":20,"cache_read_input_tokens":1100,"output_tokens":1}'

line '{"type":"system","subtype":"init","session_id":"s1"}'
ev '{"type":"stream_event","session_id":"s1","parent_tool_use_id":null,"event":{"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":'"$U1"'}}}'
ev '{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}}'
ev '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}'
ev '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}}'
line '{"type":"assistant","session_id":"s1","parent_tool_use_id":null,"message":{"id":"msg_1","model":"claude-x","role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"}],"usage":'"$U1"'}}'
ev '{"type":"stream_event","event":{"type":"content_block_stop","index":0}}'
ev '{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}}'
ev '{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me look."}}}'
line '{"type":"assistant","session_id":"s1","parent_tool_use_id":null,"message":{"id":"msg_1","model":"claude-x","role":"assistant","content":[{"type":"text","text":"Let me look."}],"usage":'"$U1"'}}'
ev '{"type":"stream_event","event":{"type":"content_block_stop","index":1}}'
ev '{"type":"stream_event","event":{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}}}'
ev '{"type":"stream_event","event":{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/x\"}"}}}'
line '{"type":"assistant","session_id":"s1","parent_tool_use_id":null,"message":{"id":"msg_1","model":"claude-x","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"/x"}}],"usage":'"$U1"'}}'
ev '{"type":"stream_event","event":{"type":"content_block_stop","index":2}}'
ev '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}}'
ev '{"type":"stream_event","event":{"type":"message_stop"}}'
line '{"type":"user","session_id":"s1","parent_tool_use_id":null,"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"x"}]}}'
ev '{"type":"stream_event","session_id":"s1","parent_tool_use_id":null,"event":{"type":"message_start","message":{"id":"msg_2","model":"claude-x","usage":'"$U2"'}}}'
ev '{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}'
ev '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Found it."}}}'
line '{"type":"assistant","session_id":"s1","parent_tool_use_id":null,"message":{"id":"msg_2","model":"claude-x","role":"assistant","content":[{"type":"text","text":"Found it."}],"usage":'"$U2"'}}'
ev '{"type":"stream_event","event":{"type":"content_block_stop","index":0}}'
ev '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}}'
ev '{"type":"stream_event","event":{"type":"message_stop"}}'
line '{"type":"result","subtype":"success","is_error":false,"session_id":"s1","result":"Found it.","total_cost_usd":0.01,"num_turns":2,"duration_ms":100,"usage":{"input_tokens":30,"cache_read_input_tokens":2100,"output_tokens":35}}'
