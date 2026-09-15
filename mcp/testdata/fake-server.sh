#!/bin/sh
# A minimal MCP server over stdio: the handshake, a paginated tool list, a call
# returning mixed content, and the server-initiated frames a client must step
# over. The request id is echoed back, as the protocol requires.
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p')
  case "$line" in
    *'"server/discover"'*)
      # A server of the handshake era does not know this request. Refusing it
      # is what tells a client which era it has reached.
      printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32601,"message":"unknown method"}}\n' "$id"
      ;;
    *'"initialize"'*)
      printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info"}}'
      printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"fake","version":"9.9"},"instructions":"be careful"}}\n' "$id"
      ;;
    *'"tools/list"'*)
      case "$line" in
        *'"cursor":"page2"'*)
          printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"second","description":"page two","inputSchema":{"type":"object"}}]}}\n' "$id"
          ;;
        *)
          printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"first","description":"page one","inputSchema":{"type":"object"}}],"nextCursor":"page2"}}\n' "$id"
          ;;
      esac
      ;;
    *'"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":"%s","result":{"content":[{"type":"text","text":"done"},{"type":"image","mimeType":"image/png","data":"aGk="}],"isError":false}}\n' "$id"
      ;;
    *'"notifications/initialized"'*) ;;
  esac
done
