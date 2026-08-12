from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from anonymize import anonymize_text
from rag import answer_about_student

agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [anonymize_text, answer_about_student]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant with two tools: one anonymizes "
        "locations and dates in a piece of text, another answers "
        "questions about the student Mariana. Choose the right tool "
        "based on what the user is asking for.",
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(agent_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_cli() -> None:
    """Interactive terminal loop for the agent — type 'exit' or 'quit' to stop."""
    print("NER Anonymization & Student RAG Agent — type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = agent_executor.invoke({"input": user_input})
        print("Agent:", result["output"], "\n")


if __name__ == "__main__":
    run_cli()