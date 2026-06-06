from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from esquemas import EvaluacionEntrevista
from src.config import PROMPT_EVALUACION_PATH, cargar_prompt


def evaluar_entrevista(transcripcion: str):
    """Evalúa una transcripción de entrevista técnica utilizando ChatGroq y un esquema estructurado."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    system_prompt = cargar_prompt(PROMPT_EVALUACION_PATH)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{transcripcion}")
    ])

    structured_llm = llm.with_structured_output(
        EvaluacionEntrevista
    )

    chain = prompt | structured_llm

    resultado = chain.invoke({
        "transcripcion": transcripcion
    })

    return resultado
