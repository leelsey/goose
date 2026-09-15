#!/bin/sh
# Refuses the discovery probe the way a server of the handshake era does,
# answers the handshake, and then goes quiet the way a tool server that has
# wedged behaves: the pipe stays open and nothing else is ever written.
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p')
  case "$line" in
    *'"server/discover"'*)
      printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32601,"message":"unknown method"}}\n' "$id"
      ;;
    *'"initialize"'*)
      printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"wedged"}}}\n' "$id"
      ;;
    *'"notifications/initialized"'*) ;;
    *)
      while true; do sleep 3600; done
      ;;
  esac
done
