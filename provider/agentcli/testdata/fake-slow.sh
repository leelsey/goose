#!/bin/sh
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"start"}}}'
sleep 30
printf '%s\n' '{"type":"result","subtype":"success"}'
