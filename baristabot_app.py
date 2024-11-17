import os
from dotenv import load_dotenv
import streamlit as st
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolMessage
from random import randint
from typing import Annotated
from typing_extensions import TypedDict

# Load .env file
load_dotenv()

# Set up Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found! Add it to a .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# ----- Define BaristaBot State and Functions -----
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool


# Define constants
BARISTABOT_SYSINT = (
    "system",
    "You are BaristaBot, a virtual barista to help users order from the cafe menu. Guide them through selecting drinks "
    "and modifiers, confirming orders, and placing them. Provide polite and helpful responses."
)

MENU = """
MENU:
Coffee Drinks:
Espresso, Americano, Cold Brew, Latte, Cappuccino, Mocha

Modifiers:
Milk: Whole, Oat, Almond
Sweeteners: Vanilla, Hazelnut, Caramel
"""

@tool
def get_menu() -> str:
    """Retrieve the café menu."""
    return MENU


@tool
def add_to_order(drink: str, modifiers: list[str]) -> str:
    """Add a drink with specified modifiers to the order."""
    return f"Added {drink} with {', '.join(modifiers) if modifiers else 'no modifiers'}."


@tool
def confirm_order(order: list[str]) -> str:
    """Confirm the user's current order."""
    return f"Your current order is: {', '.join(order)}."


@tool
def clear_order() -> str:
    """Clear all items from the user's order."""
    return "Order cleared."


@tool
def place_order(order: list[str]) -> int:
    """Place the order and return an estimated wait time."""
    return randint(1, 5)


# Order management logic
def order_node(state: OrderState) -> OrderState:
    tool_msg = state["messages"][-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":
            drink = tool_call["args"]["drink"]
            modifiers = tool_call["args"]["modifiers"]
            order.append(f"{drink} ({', '.join(modifiers) if modifiers else 'no modifiers'})")
            response = f"Order updated: {', '.join(order)}."

        elif tool_call["name"] == "confirm_order":
            response = f"Your order is: {', '.join(order)}. Is this correct?"

        elif tool_call["name"] == "clear_order":
            order.clear()
            response = "Order cleared."

        elif tool_call["name"] == "place_order":
            order_placed = True
            response = f"Order placed! ETA: {randint(1, 5)} minutes."

        else:
            response = "Unrecognized action."
        
        outbound_msgs.append(ToolMessage(content=response, name=tool_call["name"], tool_call_id=tool_call["id"]))

    return {"messages": outbound_msgs, "order": order, "finished": order_placed}


# Graph setup
tools = [get_menu, add_to_order, confirm_order, clear_order, place_order]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(OrderState)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm_with_tools.invoke([BARISTABOT_SYSINT] + state["messages"])]})
graph_builder.add_node("ordering", order_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", lambda state: "ordering" if hasattr(state["messages"][-1], "tool_calls") else "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")
graph_builder.add_edge(START, "chatbot")
chat_graph = graph_builder.compile()

# Streamlit app
def main():
    st.title("BaristaBot Café")

    if "state" not in st.session_state:
        st.session_state["state"] = {"messages": [], "order": [], "finished": False}

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input:
        st.session_state["state"]["messages"].append(("user", user_input))
        st.session_state["state"] = chat_graph.invoke(st.session_state["state"])

    for msg in st.session_state["state"]["messages"]:
        role = "You" if msg["name"] == "user" else "Bot"
        st.write(f"{role}: {msg.content}")

    if st.session_state["state"]["finished"]:
        st.success("Thank you for your order!")

if __name__ == "__main__":
    main()
