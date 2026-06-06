from typing import TypedDict, Annotated
from operator import add
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.config import PROMPT_RECLUTADOR_PATH, cargar_prompt


class EstadoEntrevista(TypedDict):
    messages: Annotated[list, add]
    nombre: str
    anios_experiencia: int
    resumen_cv: str
    contexto_documentos: str
    numero_pregunta: int

_graph_cache = None


def construir_grafo(forzar=False):
    """Construye, compila y guarda en caché el grafo de LangGraph para la entrevista."""
    global _graph_cache
    if _graph_cache is not None and not forzar:
        return _graph_cache

    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    # Prompt
    system_prompt = cargar_prompt(PROMPT_RECLUTADOR_PATH)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    # Nodo
    def nodo_entrevista(state: EstadoEntrevista):
        chain = prompt | llm
        respuesta = chain.invoke({
            "nombre": state["nombre"],
            "anios_experiencia": state["anios_experiencia"],
            "resumen_cv": state["resumen_cv"],
            "contexto_documentos": state["contexto_documentos"],
            "messages": state["messages"]
        })
        return {
            "messages": [respuesta],
            "numero_pregunta": state["numero_pregunta"] + 1,
        }

    workflow = StateGraph(EstadoEntrevista)
    workflow.add_node("entrevista", nodo_entrevista)
    workflow.set_entry_point("entrevista")
    workflow.add_edge("entrevista", END)

    memory = MemorySaver()
    _graph_cache = workflow.compile(
        checkpointer=memory
    )
    return _graph_cache
