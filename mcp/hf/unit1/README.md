** Core Primitives Implementation**

https://huggingface.co/learn/mcp-course/unit1/sdk
`basic-server.py`

```
mcp dev basic-server.py

open http://127.0.0.1:6274
```


** Use playwright mcp **

https://huggingface.co/learn/mcp-course/unit1/mcp-clients
`agents.json`

```
tiny-agents run agent.json
```

Sample prompt:

> Do a Web Search for HF inference providers on Brave Search and open the first result and then give me the list of the inference providers supported on Hugging Face



**  Gradio MCP Integration **

https://huggingface.co/learn/mcp-course/unit1/gradio-mcp
`gradio-server.py`

- <http://127.0.0.1:7860>
- <http://127.0.0.1:7860/gradio_api/mcp/>


```
tiny-agents run gradio-agent.json
```

