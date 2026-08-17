from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

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

SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

async def ask_agent(user_input: str) -> str:
    """Opens a fresh MCP connection for this single request and closes it
    afterwards — simpler and more robust than keeping one long-lived
    connection alive across python-telegram-bot's per-message tasks."""
    server_params = StdioServerParameters(command="python", args=[SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_tool_calling_agent(agent_llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            result = await agent_executor.ainvoke({"input": user_input})
            return result["output"]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I can anonymize locations/dates in text, or answer questions "
        "about the student Mariana. Just send me a message."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    output = await ask_agent(update.message.text)
    await update.message.reply_text(output)



def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running... Now you can send a message to NER assistant\n")
    app.run_polling()


if __name__ == "__main__":
    main()

