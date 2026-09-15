#!/bin/sh
# A server of the handshake era that says nothing at all to a request it does
# not know, which is what the specification warns a probing client to expect.
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p')
  case "$line" in
    *'"initialize"'*)
      printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"silent"}}}\n' "$id"
      ;;
    *'"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"only","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
  esac
done
