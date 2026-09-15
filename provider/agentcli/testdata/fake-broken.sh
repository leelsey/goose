#!/bin/sh
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}}'
printf '%s\n' 'this is not json'
