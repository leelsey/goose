#!/bin/sh
printf '%s\n' '{"type":"thread.started","thread_id":"thread-7"}'
printf '%s\n' '{"type":"item.completed","item":{"id":"i1","type":"reasoning","text":"thinking it through"}}'
printf '%s\n' '{"type":"item.completed","item":{"id":"i2","type":"agent_message","text":"done"}}'
printf '%s\n' '{"type":"turn.completed","usage":{"input_tokens":90,"cached_input_tokens":50,"output_tokens":12,"reasoning_output_tokens":4}}'
