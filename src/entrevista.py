import pandas as pd
from langchain_core.messages import HumanMessage
from src.config import TOP_CANDIDATOS_CSV_PATH, REPORTES_DIR
from src.grafo import construir_grafo
from src.transcripcion import construir_transcripcion
from src.evaluador import evaluar_entrevista
from src.reporte import generar_reporte_markdown


def realizar_entrevista_cli(candidato, retriever):
    """Realiza una entrevista a un único candidato a través de la terminal."""
    graph = construir_grafo()

    print("\n" + "=" * 60)
    print(f"ENTREVISTA (CLI): {candidato['nombre']}")
    print("=" * 60)

    consulta_rag = f"Preguntas técnicas para {candidato['resumen_cv']}"
    documentos = retriever.invoke(consulta_rag)
    contexto = "\n\n".join([doc.page_content for doc in documentos])

    thread_id = f"sesion_{candidato['id_candidato']}"
    entrada_usuario = "Hola, estoy listo para comenzar."
    numero_pregunta = 0
    respuesta_state = None

    while True:
        if numero_pregunta >= 3 or entrada_usuario.lower() == "salir":
            print(f"\nFinalizando entrevista de {candidato['nombre']}...")
            
            estado_final = graph.get_state(config={"configurable": {"thread_id": thread_id}})
            messages = estado_final.values.get("messages", []) if estado_final.values else []
            
            if not messages and respuesta_state and "messages" in respuesta_state:
                messages = respuesta_state["messages"]

            if messages:
                print("Generando reporte de la entrevista...")
                transcripcion = construir_transcripcion(messages)
                evaluacion = evaluar_entrevista(transcripcion)
                reporte = generar_reporte_markdown(candidato, evaluacion)
                
                nombre_archivo = f"{candidato['nombre']}.md"
                ruta_reporte = REPORTES_DIR / nombre_archivo
                with open(ruta_reporte, "w", encoding="utf-8") as f:
                    f.write(reporte)
                print(f"Reporte guardado exitosamente en: {ruta_reporte}")
                return nombre_archivo
            else:
                print("No se encontraron mensajes para evaluar.")
                return None

        estado = {
            "messages": [
                HumanMessage(content=entrada_usuario)
            ],
            "nombre": candidato["nombre"],
            "anios_experiencia": int(candidato["anios_experiencia"]),
            "resumen_cv": candidato["resumen_cv"],
            "contexto_documentos": contexto,
            "numero_pregunta": numero_pregunta
        }

        respuesta_state = graph.invoke(
            estado,
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        ultimo_mensaje = respuesta_state["messages"][-1].content
        print(f"\n> Reclutador:\n{ultimo_mensaje}")
        entrada_usuario = input("\n> Tu: ")
        numero_pregunta += 1


def iniciar_entrevistas_interactivas(
    archivo_candidatos=TOP_CANDIDATOS_CSV_PATH,
    retriever=None
):
    """Inicia la entrevista del primer candidato de la lista (1 iteración con 3 preguntas)."""
    if retriever is None:
        print("Error: Se requiere un retriever de base de conocimiento (RAG).")
        return

    print("\n2. Iniciando entrevista interactiva única...\n")
    try:
        df = pd.read_csv(archivo_candidatos)
        if df.empty:
            print("El archivo de candidatos está vacío.")
            return
        candidato = df.iloc[0].to_dict()
        if "id_candidato" not in candidato:
            candidato["id_candidato"] = "CAND_001"
    except FileNotFoundError:
        print(f"No se encontró el archivo de candidatos: {archivo_candidatos}")
        return

    realizar_entrevista_cli(candidato, retriever)
