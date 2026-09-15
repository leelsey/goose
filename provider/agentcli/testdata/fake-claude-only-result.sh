#!/bin/sh
# A build whose frames goose cannot read: only the closing line carries the
# answer, and it is fully billed either way.
cat >/dev/null
printf '%s\n' '{"type":"system","subtype":"init","session_id":"s1"}'
printf '%s\n' '{"type":"result","session_id":"s1","subtype":"success","result":"late answer","total_cost_usd":0.001,"duration_ms":90,"usage":{"input_tokens":7,"output_tokens":3}}'
