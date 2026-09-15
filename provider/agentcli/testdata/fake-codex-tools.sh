#!/bin/sh
# An agentic run that ends in tool work: the answer is the tool it ran, and the
# turn is not finished until the tool has answered.
cat >/dev/null
printf '%s\n' '{"type":"thread.started","thread_id":"t1"}'
printf '%s\n' '{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"checking"}}'
printf '%s\n' '{"type":"item.completed","item":{"id":"i2","type":"command_execution","command":"ls"}}'
printf '%s\n' '{"type":"turn.completed","usage":{"input_tokens":9,"output_tokens":2}}'
