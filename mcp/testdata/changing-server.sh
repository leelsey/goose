#!/bin/sh
# Offers one tool at first and two after, the way a server whose tool set
# changes while the proxy runs behaves.
count=0
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
  case "$line" in
    *'"server/discover"'*)
      printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32601,"message":"unknown method"}}\n' "$id"
      ;;
    *'"initialize"'*)
      printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"changing"}}}\n' "$id"
      ;;
    *'"tools/list"'*)
      count=$((count + 1))
      if [ "$count" -le 1 ]; then
        printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"first","inputSchema":{"type":"object"}}]}}\n' "$id"
      else
        printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"first","inputSchema":{"type":"object"}},{"name":"second","inputSchema":{"type":"object"}}]}}\n' "$id"
      fi
      ;;
  esac
done
