# Sistema de Reclutamiento Inteligente

## Requisitos del Sistema

### 1. Instalar Python 3.10 o superior

Descargar desde: <https://www.python.org/downloads/>

Verificar instalación:

```powershell
python --version
```

### 2. Crear y activar entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar dependencias

```powershell
pip install uvicorn fastapi pydantic pandas scikit-learn markdown matplotlib seaborn
pip install langchain-groq langchain-core langchain-community
pip install langchain-text-splitters langchain-chroma langchain-huggingface
pip install langgraph sentence-transformers python-multipart
pip install matplotlib
pip install seaborn
```

> **Nota**: La primera vez que se ejecute el RAG, descargará automáticamente el modelo de embeddings `all-MiniLM-L6-v2` (~80 MB).

### 4. Ejecutar la aplicación

```powershell
python main.py
```

Esto mostrará un menú interactivo:

1. **Generar base de datos** — crea `candidatos.csv` con 1000 candidatos ficticios.
2. **Filtrar candidatos** — selecciona el Top 5 y genera `top_candidatos_v2.csv`.
3. **Iniciar Portal Web** — arranca FastAPI en `http://127.0.0.1:8000`.
4. **Iniciar entrevista CLI** — entrevista interactiva por terminal.
5. **Salir**

Flujo recomendado: `1 → 2 → 3`

### 5. Estructura del proyecto

```powershell
SistemaReclutamiento/
├── main.py                  # Punto de entrada (menú CLI)
├── REQUISITOS.md            # Este archivo
├── src/
│   ├── app.py               # Servidor web FastAPI
│   ├── config.py            # Configuración y rutas
│   ├── datos.py             # Generación de candidatos ficticios
│   ├── filtro.py            # Filtrado y scoring de candidatos
│   ├── graficos.py          # Visualización de métricas KNN
│   ├── database.py          # Base de datos local (JSON)
│   ├── grafo.py             # Grafo LangGraph para entrevistas
│   ├── rag.py               # RAG con PDFs (LangChain + Chroma)
│   ├── entrevista.py        # Entrevista por CLI
│   ├── transcripcion.py     # Construcción de transcripciones
│   ├── evaluador.py         # Evaluación de entrevistas con LLM
│   └── reporte.py           # Generación de reportes Markdown
├── esquemas/
│   └── esquema_evaluacion.py  # Esquemas Pydantic
├── prompts/
│   ├── prompt_reclutador.md   # Prompt del entrevistador
│   └── evaluacion_final.md    # Prompt de evaluación
├── conocimiento/              # PDFs subidos para RAG
├── graficos/                  # Gráficos de métricas KNN
├── reportes/                  # Reportes generados
├── candidatos.csv             # Datos generados (opción 1)
├── top_candidatos_v2.csv      # Candidatos filtrados (opción 2)
└── datos_entrevistas.json     # Estado de entrevistas
```

### 6. Solución de problemas

| Problema | Solución |

|----------|----------|
| Puerto 8000 ocupado | El menú lo libera automáticamente. Si falla, ejecutar: `netstat -ano \| findstr :8000` y matar el PID con `taskkill /F /PID <PID>` |
| Error "No module named 'langchain_*'" | Ejecutar `pip install` nuevamente o verificar que el entorno virtual esté activo |
| Error de API key de Groq | Verificar `$env:GROQ_API_KEY` o revisar `src/config.py` |
| No se generan candidatos | Ejecutar opción 1 del menú |

### 7. Dependencias principales

| Librería | Uso |

|----------|-----|
| FastAPI + Uvicorn | Servidor web |
| Pandas | Procesamiento de datos |
| scikit-learn | Normalización y scoring |
| matplotlib + seaborn | Visualización de métricas del modelo |
| LangChain + LangGraph | Pipeline de entrevistas con IA |
| Chroma + HuggingFace | RAG sobre PDFs |
| Groq (Llama 3.3) | LLM para entrevistas y evaluación |
