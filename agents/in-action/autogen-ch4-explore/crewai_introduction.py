import sys
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()

joke_researcher = Agent(
    role="Senior Joke Researcher",
    goal="Research what makes things funny about the following {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "Driven by slapstick humor, you are a seasoned joke researcher "
        "who knows what makes people laugh. You have a knack for finding "
        "the funy in everyday situations and can turn a dull moment into "
        "a laugh riot."
        ),
    allow_delegation=True,
    )

joke_writer = Agent(
    role="Joke Writer",
    goal="Write a humourous and funny joke on the following {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "You are a joke writer with a flair for huor. You can turn a "
        "simple idea into a laugh riot. You have a way with words and "
        "can make people laugh with just a few lines."
        ),
    allow_delegation=True,
    )

research_task = Task(
    description=(
        "Identify what makes the following topic: {topic} so funny. "
        "Be sure to include the key elements that make it humourous."
        "Also, provide an analysis of the current social trends, "
        "and how it impacts the perception of humor."
        ),
    expected_output="A comprehensive 3 paragraphs long report on the latest jokes.",
    agent=joke_researcher,
    )

write_task = Task(
    description=(
        "Compose an insightful, huomourous and socially aware joke on {topic}. "
        "Be sure to include the key elements that make it funny and "
        "relevant to the current social trends."
        ),
    expected_output="A joke on {topic}.",
    agent=joke_writer,
    async_execution=False,
    output_file="the_best_joke.md",
    )

crew = Crew(
    agents=[joke_researcher, joke_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
    )

topic = " ".join(sys.argv[1:])
print(f"Starting for: >>{topic}<<...")
assert topic, "Topic should not be empty"
result = crew.kickoff(inputs={"topic": topic})
print("=== Done")
print(result)

