import sys
import subprocess
import uvicorn
from src.datos import ejecutar_generacion
from src.filtro import ejecutar_filtro_personalizado
from src.rag import preparar_base_conocimiento_rag
from src.entrevista import iniciar_entrevistas_interactivas
from src.database import inicializar_db
from src.config import RAG_DIR, TOP_CANDIDATOS_CSV_PATH


def liberar_puerto(port):
    """Mata el proceso que ocupa el puerto dado, si existe."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                pid = parts[4]
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True, timeout=5)
                print(f"Puerto {port} liberado (se detuvo el proceso {pid}).")
                return
    except Exception as e:
        print(f"No se pudo liberar el puerto {port}: {e}")


def menu():
    """Muestra el menú interactivo para coordinar el sistema de reclutamiento."""
    while True:
        print("\n" + "=" * 50)
        print("      SISTEMA DE RECLUTAMIENTO INTELIGENTE      ")
        print("=" * 50)
        print("1. Generar base de datos de candidatos ficticios (candidatos.csv)")
        print("2. Filtrar y puntuar candidatos (top_candidatos_v2.csv)")
        print("3. Iniciar Portal Web del Reclutador (FastAPI en http://127.0.0.1:8000)")
        print("4. Iniciar entrevista de prueba en Terminal (CLI)")
        print("5. Salir")
        print("=" * 50)

        opcion = input("Selecciona una opción (1-5): ").strip()

        if opcion == "1":
            print("\nGenerando candidatos...")
            ejecutar_generacion()
        elif opcion == "2":
            print("\nFiltrando candidatos...")
            ejecutar_filtro_personalizado()
            print("Inicializando base de datos para la interfaz web...")
            inicializar_db(forzar=True)
        elif opcion == "3":
            print("\nIniciando Portal Web del Reclutador...")
            # Liberar puerto si está ocupado
            liberar_puerto(8000)
            print("\nEl servidor estará disponible en: http://127.0.0.1:8000")
            print("Presiona Ctrl+C para detener el servidor.\n")
            try:
                # Arrancar FastAPI usando Uvicorn
                uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
            except KeyboardInterrupt:
                print("\nServidor web detenido.")
        elif opcion == "4":
            print("\nIniciando entrevista interactiva en consola...")
            if not TOP_CANDIDATOS_CSV_PATH.exists():
                print(f"\n[!] Error: No se encontró el archivo de candidatos filtrados ({TOP_CANDIDATOS_CSV_PATH}).")
                print("Por favor, ejecuta la opción 2 primero.")
                continue

            pdfs = list(RAG_DIR.glob("*.pdf"))
            if not pdfs:
                print(f"\n[!] Error: No hay PDFs de conocimiento en '{RAG_DIR}'.")
                print("Sube archivos PDF desde el portal web (opción 3).")
                continue

            retriever = preparar_base_conocimiento_rag(RAG_DIR)
            if retriever:
                iniciar_entrevistas_interactivas(TOP_CANDIDATOS_CSV_PATH, retriever)
            else:
                print("\n[!] Error: No se pudo configurar el recuperador RAG.")
        elif opcion == "5":
            print("\n¡Gracias por usar el Sistema de Reclutamiento!")
            sys.exit(0)
        else:
            print("\nOpción no válida. Por favor, ingresa un número de 1 a 5.")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido. ¡Adiós!")
        sys.exit(0)
