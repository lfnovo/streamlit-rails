import streamlit as st
from langchain_core.messages import HumanMessage
from typing import Type, TypeVar, Dict, Any, Union
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from .langgraph_utils import make_config, convert_to_model
from typing_extensions import TypedDict, get_type_hints

T = TypeVar('T', bound=BaseModel)

def is_typeddict(obj):
    try:
        return isinstance(obj, dict) and hasattr(obj.__class__, "__annotations__")
    except AttributeError:
        return False


def run_thread(
    user_answer: str,
    state_key: str,
    state_class: Type[T],
    graph: Runnable,
    thread_id: str,
) -> None:
    """Run the graph with a user answer.
    
    Args:
        user_answer: The user's input message
        state_key: Key to access the state in st.session_state
        state_class: The class type for the state (e.g. ThreadState)
        graph: The LangGraph instance to use
        thread_id: The thread identifier
    """
    was_dict = isinstance(st.session_state[state_key], dict)
    if not was_dict:
        model_state = convert_to_model(st.session_state[state_key], state_class)
    else:
        model_state = st.session_state[state_key]

    model_dict = model_state if was_dict else model_state.model_dump()
    
    if not model_dict.get("messages", []):
        model_dict["messages"] = []
    model_dict["messages"].append(HumanMessage(content=user_answer))

    st.session_state[state_key] = convert_to_model(graph.invoke(
        input=model_dict, 
        config=make_config(thread_id)
    ), state_class)


def chat_component(
    state_key: str,
    state_class: Type[T],
    graph: Runnable,
    thread_id: str,
    initial_state: Union[Dict, BaseModel] = None,
    placeholder: str = "What do you want to learn?"
) -> None:
    """Generic chat interface component.
    
    Args:
        state_key: Key to access the state in st.session_state
        state_class: The class type for the state (e.g. ThreadState)
        graph: The LangGraph instance to use
        thread_id: The thread identifier
        initial_state: Initial state to use if state_key doesn't exist in session_state
        placeholder: The placeholder text for the chat input
    """
    # Inicializa o state no session_state se não existir
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_state or state_class()

    
    if prompt := st.chat_input(placeholder):
        run_thread(prompt, state_key, state_class, graph, thread_id)

    was_dict = isinstance(st.session_state[state_key], dict)
    if not was_dict:
        model_state = convert_to_model(st.session_state[state_key], state_class)
    else:
        model_state = st.session_state[state_key]

    model_dict = model_state if was_dict else model_state.model_dump()


    messages = model_dict.get("messages", [])
    for message in messages[::-1]:
        message_type = message.type if not isinstance(message, dict) else message.get("type")
        message_content = message.content if not isinstance(message, dict) else message.get("content")
        with st.chat_message(message_type):
            st.markdown(message_content)
