import os
from autogen import ConversableAgent, UserProxyAgent

def is_termination_msg(x):
    return x.get("content", "").rstrip().endswith("TERMINATE")

assistant = ConversableAgent(
    name="agent",
    llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}]},
    is_termination_msg=is_termination_msg,
)

user_proxy = UserProxyAgent(
    name="user",
    code_execution_config={
        "work_dir": "tmp",
        "use_docker": False,
    },
    human_input_mode="ALWAYS",
    is_termination_msg=is_termination_msg,
)

user_proxy.initiate_chat(assistant, message="Write a solution for fizz buzz in one line")
