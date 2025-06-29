from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, OpenAIServerModel

import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
model = OpenAIServerModel(
        model_id = "gpt-4o-mini",
        api_key = OPENAI_API_KEY
        )

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
