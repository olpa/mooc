import os
from autogen import AssistantAgent, UserProxyAgent

llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}]}


user_proxy = UserProxyAgent(
    name="user",
    code_execution_config={
        "work_dir": "tmp",
        "use_docker": False,
        "last_n_messages": 1,
    },
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)


engineer = AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message="""
    You are a professional Python engineer, known for your expertise in software development.
    You use your skills to create software applications, tools, and games that are both functional and efficient.
    Your preference is to write clean, well-structured code that is easy to read and maintain.
    """,
    )

critic = AssistantAgent(
    name="Reviewer",
    llm_config=llm_config,
    system_message="""
    You are a code reviewer, known for your throughness and commitment to standards.
    Your task is to scutinize code content for any harmful or substandard elements.
    You ensure that the code is secure, efficient, and adheres to best practices.
    """,
)


def review_code(recipient, messages, sender, config):
    return f"""
    Review and critique the following code.

    {messages[-1]['content']}
    """

user_proxy.register_nested_chats(
    [
        {
            "recipient": critic,
            "message": review_code,
            "summary_method": "last_msg",
            "max_turns": 1,
        }
    ],
    trigger=engineer,
)

#task = """Write a snake game using Pygame."""
task = """Write a fibonacci function for nth number."""

res = user_proxy.initiate_chat(
    recipient=engineer,
    message=task,
    max_turns=4,
    summary_method="last_msg"
)

