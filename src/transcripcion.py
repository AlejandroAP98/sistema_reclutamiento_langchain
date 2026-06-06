from langchain_core.messages import HumanMessage, AIMessage


def construir_transcripcion(messages):
    """Construye un string legible de transcripción a partir de una lista de mensajes."""
    lineas = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            lineas.append(f"CANDIDATO: {msg.content}")
        elif isinstance(msg, AIMessage):
            lineas.append(f"RECLUTADOR: {msg.content}")

    return "\n\n".join(lineas)
