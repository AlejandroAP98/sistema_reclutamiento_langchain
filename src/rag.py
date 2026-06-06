from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import RAG_DIR


def preparar_base_conocimiento_rag(ruta=RAG_DIR):
    """Carga todos los PDFs de un directorio y genera un recuperador vectorial."""
    if isinstance(ruta, (str, Path)):
        ruta = Path(ruta)

    if ruta.is_file():
        pdfs = [ruta]
    elif ruta.is_dir():
        pdfs = list(ruta.glob("*.pdf"))
    else:
        print(f"Ruta no válida: {ruta}")
        return None

    if not pdfs:
        print("No se encontraron archivos PDF.")
        return None

    print(f"Cargando {len(pdfs)} PDF(s) de conocimiento...")
    documentos = []
    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = pdf.name
            documentos.extend(docs)
        except Exception as e:
            print(f"Error al cargar {pdf.name}: {e}")

    if not documentos:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})
