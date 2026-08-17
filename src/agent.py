from dotenv import load_dotenv
load_dotenv()

import asyncio
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

# Path to our MCP server script, so the client knows what process to start
SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")


async def run_cli() -> None:
    """Interactive terminal loop for the agent — type 'exit' or 'quit' to stop.

    Connects to our MCP server over stdio, loads its tools as LangChain
    tools, and builds the agent fresh for this session.
    """
    server_params = StdioServerParameters(command="python", args=[SERVER_PATH])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_tool_calling_agent(agent_llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            print("NER Anonymization & Student RAG Agent — type 'exit' to quit.\n")
            while True:
                try:
                    user_input = input("You: ")
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break

                if user_input.strip().lower() in ("exit", "quit", "exit()", "q"):
                    print("Goodbye!")
                    break

                result = await agent_executor.ainvoke({"input": user_input})
                print("Agent:", result["output"], "\n")


if __name__ == "__main__":
    asyncio.run(run_cli())