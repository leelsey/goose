#!/bin/sh
# Records how it was invoked so the argv and the environment can be asserted.
{
  printf 'ARGS'
  for a in "$@"; do printf '\t%s' "$a"; done
  printf '\n'
  printf 'STDIN\t%s\n' "$(cat 2>/dev/null | head -c 200 | tr '\n' ' ')"
  printf 'ENV\t%s\n' "$(env | sort | tr '\n' ' ')"
} >> "$GOOSE_TEST_RECORD"
printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}}'
printf '%s\n' '{"type":"result","subtype":"success"}'
