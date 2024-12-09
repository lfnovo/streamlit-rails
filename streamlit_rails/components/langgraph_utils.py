from langchain_core.runnables import RunnableConfig
from typing import Type, TypeVar, Dict, Any, Union
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

def convert_to_model(state: Union[Dict, BaseModel, Any], state_class: Type[T]) -> T:
    """Convert a state to the appropriate model type.
    
    Args:
        state: The state to convert (can be dict, AddableValuesDict, or the model itself)
        state_class: The target state class
    
    Returns:
        T: An instance of the state class
    """ 
    # Se for um dicionário (incluindo TypedDict), converte diretamente
    if isinstance(state, dict):
        return state_class(**state)
    # Se tiver __dict__, converte para dicionário
    elif hasattr(state, "__dict__"):
        return state_class(**dict(state))
    # Se for o próprio tipo que queremos, retorna direto
    elif type(state) == state_class:
        return state
    else:
        # Se não tiver state, cria um novo
        return state_class()


def make_config(thread_id: str) -> RunnableConfig:
    """Get the graph configuration with the current thread ID."""
    return {"configurable": {"thread_id": thread_id}}


