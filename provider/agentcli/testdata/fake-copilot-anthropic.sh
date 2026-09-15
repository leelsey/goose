#!/bin/sh
# This binary fronts the Anthropic API and relays its framing unchanged on some
# builds: without those names the whole answer is dropped.
cat >/dev/null
printf '%s\n' '{"type":"message_start","session_id":"s1"}'
printf '%s\n' '{"type":"content_block_start","content_block":{"type":"text","text":""}}'
printf '%s\n' '{"type":"content_block_delta","delta":{"type":"text_delta","text":"copilot "}}'
printf '%s\n' '{"type":"content_block_delta","delta":{"type":"text_delta","text":"answered"}}'
printf '%s\n' '{"type":"message_delta","usage":{"input_tokens":5,"output_tokens":2}}'
printf '%s\n' '{"type":"message_stop"}'
