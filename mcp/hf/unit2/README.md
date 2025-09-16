# Use case: end-to-end mcp application

## Building the Gradio MCP Server

https://huggingface.co/learn/mcp-course/unit2/gradio-server
`server.py`

```
$ python server.py

* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.

🔨 Launching MCP server:
** Streamable HTTP URL: http://127.0.0.1:7860/gradio_api/mcp/
* [Deprecated] SSE URL: http://127.0.0.1:7860/gradio_api/mcp/sse
```

##  MCP Configuration Files

https://huggingface.co/learn/mcp-course/unit2/clients
`mcp.json`

```
$ tiny-agents run mcp.json
```

## MCP in Clients

### Claude Code

Symlinks doesn't work

```
~/opt/claude-code/node_modules/.bin/claude mcp add --transport sse senti http://127.0.0.1:7860/gradio_api/mcp/sse
~/opt/claude-code/node_modules/.bin/claude mcp list
# Not checked
claude mcp add --transport http senti http://127.0.0.1:7860/gradio_api/mcp/
```

Inside Claude:

```
> Do sentiment analysis of the text: "I love Claude Code".

● senti - sentiment_analysis (MCP)(text: "I love Claude Code")

╭──────────────────────────────────────────────────────────────────────────────╮
│ Tool use                                                                     │
│                                                                              │
│   senti - sentiment_analysis(text: "I love Claude Code") (MCP)               │
│   Analyze the sentiment of the given text. Returns: str: A JSON string       │
│   containing polarity, subjectivity and assessment                           │
│                                                                              │
│ Do you want to proceed?                                                      │
│ ❯ 1. Yes                                                                     │
│   2. Yes, and don't ask again for senti - sentiment_analysis commands in     │
│   /home/olpa/mooc/mcp/hf/unit2                                               │
│   3. No, and tell Claude what to do differently (esc)                        │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

● senti - sentiment_analysis (MCP)(text: "I love Claude Code")
  ⎿ {
       "polarity": 0.5,
       "subjectivity": 0.6,
       "assessment": "positive"
     }
```

