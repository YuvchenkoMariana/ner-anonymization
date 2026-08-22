from dotenv import load_dotenv

load_dotenv()

import os

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import asyncio
import uuid
from pathlib import Path

import streamlit as st
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
        "You are a helpful assistant with three tools: one anonymizes "
        "locations and dates in English text, another does the same for "
        "Ukrainian text, and a third answers questions about the student "
        "Mariana. Choose the right tool based on the input language and "
        "what the user is asking for.",
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

EXAMPLES = [
    ("🇬🇧", "Sarah moved to London on March 3rd, 2019."),
    ("🇺🇦", "Марія переїхала до Києва 5 січня 2020 року."),
    ("❓", "When was Mariana born?"),
]

FONT_SCALES = {"Normal": "16px", "Large": "19px", "Extra large": "23px"}

USER_BORDER = "#3e5778"
BOT_BORDER = "#b3bfce"

WELCOME_MESSAGE = (
    '<span class="welcome-marker" style="display:none;"></span>'
    "👋 Hello! This is your anonymization assistant. Here's what I can do:\n"
    "- Redact locations and dates from **English** text\n"
    "- Redact locations and dates from **Ukrainian** text\n"
    "- Answer questions about Mariana\n\n"
    "Just type a sentence or a question below to get started."
)


async def ask_agent(user_input: str) -> str:
    """Opens a fresh MCP connection for this single request — same
    per-request pattern as telegram_bot.py's ask_agent, needed because
    Streamlit reruns this whole script on every user action."""
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_PATH],
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_tool_calling_agent(agent_llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)
            result = await agent_executor.ainvoke({"input": user_input})
            return result["output"]


def highlight_entities(text: str) -> str:
    """Swaps the plain [LOC]/[DATE] markers for styled badges — purely
    cosmetic, doesn't touch what the agent actually returns."""
    text = text.replace("[LOC]", '<span class="loc-tag">📍 LOC</span>')
    text = text.replace("[DATE]", '<span class="date-tag">📅 DATE</span>')
    return text


def new_chat() -> str:
    """Creates a fresh chat entry, makes it the active one, and returns its id."""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}],
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


st.set_page_config(page_title="NER Anonymizer Assistant", page_icon="🕵️", layout="centered")

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current_chat_id = None
if not st.session_state.chats:
    new_chat()
if st.session_state.current_chat_id not in st.session_state.chats:
    # Defensive guard: if session_state ever ends up pointing at a chat id
    # that no longer exists (e.g. a stale widget value after a rerun),
    # fall back to any existing chat instead of crashing.
    st.session_state.current_chat_id = next(iter(st.session_state.chats))
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "font_scale" not in st.session_state:
    st.session_state.font_scale = "Normal"

current_chat = st.session_state.chats[st.session_state.current_chat_id]

BG = "#12141a" if st.session_state.dark_mode else "#ffffff"
CARD_BG = "#1c1f27" if st.session_state.dark_mode else "#ffffff"
TEXT_COLOR = "#e8e8ec" if st.session_state.dark_mode else "#1a1a1a"
SUBTITLE_COLOR = "#9aa4b5" if st.session_state.dark_mode else "#6b7686"
INPUT_BORDER = "#e8e8ec" if st.session_state.dark_mode else USER_BORDER
FONT_SIZE = FONT_SCALES[st.session_state.font_scale]

# NOTE: the :has() selectors below target Streamlit's internal chat testids
# (stChatMessageContent + its aria-label, which encodes the role regardless
# of avatar type) — internal, not public API, could shift in a future
# Streamlit version.
st.markdown(
    f"""
    <style>
    html {{
        font-size: {FONT_SIZE} !important;
    }}
    .stApp {{
        background: {BG} !important;
        color: {TEXT_COLOR} !important;
    }}

    div[data-testid="stAppDeployButton"] {{
        display: none !important;
    }}

    .main-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: {USER_BORDER};
        margin-bottom: 0;
    }}
    .subtitle {{
        color: {SUBTITLE_COLOR};
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }}

    div[data-testid="stLayoutWrapper"] {{
        max-width: 100% !important;
        width: 100% !important;
    }}
    div[data-testid="stChatMessage"] {{
        width: fit-content;
        border: 2px solid {BOT_BORDER};
        border-radius: 14px !important;
    }}
    div[data-testid="stChatMessage"]:has(.welcome-marker) {{
        width: 100% !important;
        max-width: 100% !important;
        margin-right: 0 !important;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) {{
        border-color: {USER_BORDER} !important;
        background: rgba(62, 87, 120, 0.08) !important;
        box-shadow: 0 0 0 3px rgba(62, 87, 120, 0.16), 0 12px 30px rgba(7, 19, 48, 0.08) !important;
        margin-left: auto;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] {{
        text-align: right !important;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from assistant"]) {{
        border-color: {BOT_BORDER} !important;
        background: rgba(179, 191, 206, 0.18) !important;
        box-shadow: 0 0 0 3px rgba(179, 191, 206, 0.35), 0 12px 30px rgba(7, 19, 48, 0.08) !important;
        margin-right: auto;
    }}
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"] {{
        padding-right: 12px;
    }}

    .stButton > button {{
        background: {USER_BORDER};
        color: #ffffff;
        border: none;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(62, 87, 120, 0.25);
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background: #4f6a8f;
        color: #ffffff;
    }}

    span.loc-tag, span.date-tag {{
        padding: 3px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85em;
    }}
    span.loc-tag {{
        background: {USER_BORDER};
        color: #ffffff;
    }}
    span.date-tag {{
        background: {BOT_BORDER};
        color: #1a1a1a;
    }}

    div[data-testid="stSelectbox"] > div > div {{
        background: {CARD_BG} !important;
        border-color: {BOT_BORDER} !important;
    }}
    div[data-testid="stSelectbox"] input {{
        color: {TEXT_COLOR} !important;
    }}
    div[data-testid="stSelectbox"] input::placeholder {{
        color: {TEXT_COLOR} !important;
        opacity: 0.7;
    }}
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {{
        color: {TEXT_COLOR} !important;
    }}
    div[data-testid="stSelectboxVirtualDropdown"] {{
        background: {CARD_BG} !important;
    }}
    div[data-testid="stSelectboxVirtualDropdown"] * {{
        color: {TEXT_COLOR} !important;
    }}
    div[data-testid="stPopoverBody"] {{
        background: {CARD_BG} !important;
        color: {TEXT_COLOR} !important;
        border-color: {BOT_BORDER} !important;
    }}

    header[data-testid="stHeader"],
    div[data-testid="stAppViewContainer"],
    section[data-testid="stMain"],
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"] {{
        background: {BG} !important;
    }}
    div[data-testid="stBottom"] * {{
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"] {{
        background: {CARD_BG} !important;
        border-width: 2px !important;
        border-style: solid !important;
        border-color: {INPUT_BORDER} !important;
        border-radius: 10px !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        background: {CARD_BG} !important;
        color: {TEXT_COLOR} !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {TEXT_COLOR} !important;
        opacity: 0.6;
    }}

    button[data-testid="stChatInputSubmitButton"] {{
        background: {USER_BORDER} !important;
        color: #ffffff !important;
    }}
    button[data-testid="stChatInputSubmitButton"] svg {{
        fill: #ffffff !important;
    }}

    button[data-testid="stPopoverButton"] {{
        background: {CARD_BG} !important;
        border-color: {BOT_BORDER} !important;
    }}
    button[data-testid="stPopoverButton"] * {{
        color: {TEXT_COLOR} !important;
    }}
    button[data-testid="stPopoverButton"]:hover {{
        background: #ffffff !important;
    }}
    button[data-testid="stPopoverButton"]:hover * {{
        color: #1a1a1a !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🕵️ NER Anonymizer Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Anonymize locations/dates in English or Ukrainian, or ask about Mariana.</p>',
    unsafe_allow_html=True,
)

# ── Top toolbar (replaces the sidebar): new chat, chat history, examples, settings ──
chat_ids_ordered = list(reversed(list(st.session_state.chats.keys())))


def format_chat(chat_id: str) -> str:
    return st.session_state.chats[chat_id]["title"]


def switch_chat() -> None:
    chat_id = st.session_state.history_select
    if chat_id is not None:
        st.session_state.current_chat_id = chat_id
        st.session_state.history_select = None


def format_example(i: int) -> str:
    flag, text = EXAMPLES[i]
    return f"{flag} {text}"


def apply_example() -> None:
    i = st.session_state.example_select
    if i is not None:
        st.session_state.pending_input = EXAMPLES[i][1]
        st.session_state.example_select = None


toolbar = st.columns([1.3, 1.7, 1.7, 0.5])

with toolbar[0]:
    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

with toolbar[1]:
    st.selectbox(
        "Chat history",
        chat_ids_ordered,
        index=None,
        placeholder="History",
        format_func=format_chat,
        key="history_select",
        on_change=switch_chat,
        label_visibility="collapsed",
    )

with toolbar[2]:
    st.selectbox(
        "Examples",
        list(range(len(EXAMPLES))),
        index=None,
        placeholder="Choose an example",
        format_func=format_example,
        key="example_select",
        on_change=apply_example,
        label_visibility="collapsed",
    )

with toolbar[3]:
    with st.popover("⚙️"):
        dark = st.toggle("🌙 Dark theme", value=st.session_state.dark_mode, key="dark_mode_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        size_choice = st.select_slider(
            "🔍 Text size",
            options=list(FONT_SCALES.keys()),
            value=st.session_state.font_scale,
            key="font_scale_select",
        )
        if size_choice != st.session_state.font_scale:
            st.session_state.font_scale = size_choice
            st.rerun()

AVATARS = {"user": "🧑‍💻", "assistant": "🕵️"}

for message in current_chat["messages"]:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(highlight_entities(message["content"]), unsafe_allow_html=True)

user_input = st.chat_input("Type something...")
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

if user_input:
    if current_chat["title"] == "New chat":
        current_chat["title"] = user_input if len(user_input) <= 30 else user_input[:27] + "..."

    current_chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            output = asyncio.run(ask_agent(user_input))
            st.markdown(highlight_entities(output), unsafe_allow_html=True)
    current_chat["messages"].append({"role": "assistant", "content": output})
