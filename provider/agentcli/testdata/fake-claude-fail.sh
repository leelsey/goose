#!/bin/sh
printf '%s\n' '{"type":"stream_event","event":{"type":"message_start","message":{"id":"m","usage":{"input_tokens":33}}}}'
printf '%s\n' '{"type":"result","subtype":"error_during_execution","is_error":true,"api_error_status":529,"result":"overloaded","usage":{"input_tokens":33},"total_cost_usd":0.0001}'
