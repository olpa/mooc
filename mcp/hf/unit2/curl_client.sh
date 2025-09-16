#!/bin/bash

# Sentiment Analysis Demo using curl with MCP senti server
# This script demonstrates how to perform sentiment analysis on text via HTTP

TEXT="I love curl"
ENDPOINT="http://127.0.0.1:7860/gradio_api/mcp/"

echo "Performing sentiment analysis on: '$TEXT'"
echo "Using endpoint: $ENDPOINT"
echo "----------------------------------------"

# Create JSON-RPC 2.0 payload for the MCP sentiment analysis request
PAYLOAD=$(cat <<EOF
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "sentiment_analysis",
    "arguments": {
      "text": "$TEXT"
    }
  }
}
EOF
)

echo "Sending request..."
echo "Payload: $PAYLOAD"
echo ""

# Send the request using curl with SSE protocol
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d "$PAYLOAD" \
  "$ENDPOINT" \
    | grep 'data:' | sed 's/data://' | jq .

