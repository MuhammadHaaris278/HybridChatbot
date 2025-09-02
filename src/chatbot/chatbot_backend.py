from typing import List, Tuple
from chatbot.load_config import LoadProjectConfig
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.build_full_graph import build_graph
from utils.app_utils import create_directory
from chatbot.memory import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load project and tools configuration
PROJECT_CFG = LoadProjectConfig()
TOOLS_CFG = LoadToolsConfig()

# Build the agent graph
graph = build_graph()
config = {"configurable": {"thread_id": TOOLS_CFG.thread_id}}

# Ensure memory directory exists
create_directory("memory")


class ChatBot:
    """
    A class to handle chatbot interactions using a pre-defined agent graph.
    The chatbot processes user messages, generates appropriate responses,
    and saves the chat history to a specified memory directory.

    Attributes:
        config (dict): A configuration dictionary with settings such as `thread_id`.

    Methods:
        respond(chatbot: List, message: str) -> Tuple:
            Processes the user message, generates a response, appends it to
            the chat history, and writes the chat history to a file.
    """

    @staticmethod
    def respond(chatbot: List, message: str) -> Tuple:
        """
        Processes a user message using the agent graph, generates a response,
        and appends it to the chat history. The chat history is also saved
        to a memory file for future reference.

        Args:
            chatbot (List): Chatbot conversation history. Each entry is a tuple of
                            (user_message, bot_response).
            message (str): The user message to process.

        Returns:
            Tuple: An empty string (for the new input placeholder) and the updated conversation history.
        """

        # Define a system prompt to guide tool usage
        system_prompt = (
            "You are a helpful assistant for retrieving clinical information. "
            "Decide which tool to use based on the query style and intent:\n"
            "- If the query is semantic or context-oriented (e.g., asking for insights, history, narrative notes, or free-text descriptions), "
            "use the lookup_clinical_notes tool to search the vector database of patient notes.\n"
            "- If the query requires structured, tabular, or coded data (e.g., diagnoses lists, lab values, medications, appointments, demographics), "
            "use the tool_hospital_sqlagent to query the hospital SQL database.\n"
            "Note that both clinical notes and structured data can cover similar topics (such as diagnoses or appointments); "
            "choose the tool based on whether the request is unstructured/semantic vs. structured/tabular.\n"
            "- For general knowledge questions unrelated to patient records, use the search_tool for web searches.\n"
            "Always provide accurate and relevant information based on the tools' outputs, "
            "and if no relevant data is found, clearly state that the information is unavailable."
        )

        # Convert Gradio history to LangChain messages and add system prompt
        messages = [SystemMessage(content=system_prompt)]
        for user_msg, bot_msg in chatbot:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))

        # Add the new user input
        messages.append(HumanMessage(content=message))

        # Pass messages into the graph
        events = graph.stream(
            {"messages": messages}, config, stream_mode="values"
        )
        for event in events:
            event["messages"][-1].pretty_print()

        # Append last response to chat history
        chatbot.append((message, event["messages"][-1].content))

        # Save conversation to memory
        Memory.write_chat_history_to_file(
            gradio_chatbot=chatbot,
            folder_path=PROJECT_CFG.memory_dir,
            thread_id=TOOLS_CFG.thread_id
        )

        return "", chatbot
