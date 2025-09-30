# Exploring multi-agent systems

## Chapter 4.1

```
export OPENAI_API_KEY=...
autogenstudio ui --port 8081
```

http://127.0.0.1:8081


For now, failed to add a tool.

## Chapter 4.2 Exploring AutoGen

`autogen_start.py`: updated to the version 0.9.

The setup of the termination message is not obvious for the first time user:

- The conversation stops when human input is "exit" OR when is_termination_msg returns True AND there's no human input
- When you type "TERMINATE", the termination condition is met, but since you're in "ALWAYS" mode, it still prompts for input
- To actually terminate, you need to either:
  a. Type "exit"
  b. Press Enter (empty input) when the termination message is detected

## Chapter 4.2.2 Enhancing code output with agent critics

`autogen_coding_critic.py`

Updated to the new version, based on

- https://microsoft.github.io/autogen/0.2/docs/notebooks/agentchat_nestedchat/
- https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat/

