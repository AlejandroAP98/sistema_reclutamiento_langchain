import json
import os
import shutil
import markdown
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

from src.config import RAG_DIR, REPORTES_DIR, CANDIDATOS_CSV_PATH, TOP_CANDIDATOS_CSV_PATH
from src.filtro import ejecutar_filtro_personalizado
from src.rag import preparar_base_conocimiento_rag
from src.grafo import construir_grafo, EstadoEntrevista
from src.transcripcion import construir_transcripcion
from src.evaluador import evaluar_entrevista
from src.reporte import generar_reporte_markdown
from src.database import (
    inicializar_db,
    obtener_candidatos,
    obtener_candidato,
    actualizar_candidato
)
from langchain_core.messages import HumanMessage

app = FastAPI(title="Recruitment System Portal")

# Variables globales para el servidor
retriever_instancia = None


@app.on_event("startup")
def startup_event():
    global retriever_instancia
    print("Iniciando servidor web...")
    # Asegurar base de datos inicializada
    inicializar_db()

    # Cargar retriever desde directorio de conocimiento
    pdfs = list(RAG_DIR.glob("*.pdf"))
    if pdfs:
        retriever_instancia = preparar_base_conocimiento_rag(RAG_DIR)
        print(f"RAG configurado con {len(pdfs)} PDF(s).")
    else:
        print(f"[!] ADVERTENCIA: No hay PDFs en '{RAG_DIR}'. RAG no estará disponible.")


class MensajeRequest(BaseModel):
    mensaje: str


class FiltroRequest(BaseModel):
    tecnologias: list[str] = []
    soft_skills: list[str] = []
    roles: list[str] = []
    remoto: bool = True
    experiencia_min: int = 2
    skill_min: int = 5
    ingles_min: str = "B2"
    top_n: int = 5


ESTILOS_COMUNES = """
    :root {
        --bg-color: #0b0f19;
        --card-bg: rgba(22, 30, 49, 0.7);
        --border-color: rgba(255, 255, 255, 0.08);
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --accent-primary: #6366f1; /* Indigo */
        --accent-secondary: #a855f7; /* Purple */
        --success: #10b981;
        --warning: #f59e0b;
        --info: #3b82f6;
    }
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    body {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
        background-attachment: fixed;
        color: var(--text-primary);
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }
    .header {
        background: rgba(11, 15, 25, 0.6);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--border-color);
        padding: 1.2rem 2rem;
        position: sticky;
        top: 0;
        z-index: 50;
    }
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .logo {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .container {
        max-width: 1200px;
        margin: 2rem auto;
        padding: 0 1.5rem;
        width: 100%;
        flex-grow: 1;
    }
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        backdrop-filter: blur(16px);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .btn {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border: none;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        text-decoration: none;
    }
    .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        opacity: 0.95;
    }
    .btn-secondary {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.15);
        box-shadow: none;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-pending {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-generated {
        background: rgba(59, 130, 246, 0.15);
        color: var(--info);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-active {
        background: rgba(99, 102, 241, 0.15);
        color: var(--accent-primary);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-completed {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
"""


STATUS_ICONS = {
    "uploaded": '<span style="color:var(--success);font-weight:700;">&#10003;</span>',
    "missing": '<span style="color:var(--warning);font-weight:700;">&#9888;</span>',
}


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser CSV.")
    with open(CANDIDATOS_CSV_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if TOP_CANDIDATOS_CSV_PATH.exists():
        TOP_CANDIDATOS_CSV_PATH.unlink()
    return JSONResponse({"status": "success", "filename": file.filename})


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever_instancia
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser PDF.")
    ruta_destino = RAG_DIR / file.filename
    with open(ruta_destino, "wb") as f:
        shutil.copyfileobj(file.file, f)
    retriever_instancia = preparar_base_conocimiento_rag(RAG_DIR)
    return JSONResponse({"status": "success", "filename": file.filename})


@app.get("/api/pdfs")
def list_pdfs():
    pdfs = []
    for p in sorted(RAG_DIR.glob("*.pdf")):
        pdfs.append({"nombre": p.name, "tamano": p.stat().st_size})
    return JSONResponse({"pdfs": pdfs})


@app.delete("/api/delete-pdf/{filename}")
def delete_pdf(filename: str):
    global retriever_instancia
    ruta = RAG_DIR / filename
    if not ruta.exists():
        raise HTTPException(status_code=404, detail="PDF no encontrado.")
    ruta.unlink()
    retriever_instancia = preparar_base_conocimiento_rag(RAG_DIR)
    return JSONResponse({"status": "deleted", "filename": filename})


@app.get("/api/filter-options")
def filter_options():
    return JSONResponse({
        "tecnologias": [],
        "soft_skills": [],
        "roles": [],
    })


@app.post("/api/filtrar")
def filtrar_candidatos(params: FiltroRequest = None):
    if not CANDIDATOS_CSV_PATH.exists():
        raise HTTPException(status_code=400, detail="Debe subir un archivo CSV primero.")

    if params is None:
        params = FiltroRequest()

    resultado = ejecutar_filtro_personalizado(
        archivo_entrada=CANDIDATOS_CSV_PATH,
        archivo_salida=TOP_CANDIDATOS_CSV_PATH,
        tecnologias_clave=params.tecnologias,
        soft_skills=params.soft_skills,
        roles_clave=params.roles,
        requiere_remoto=params.remoto,
        min_experiencia=params.experiencia_min,
        min_skill_score=params.skill_min,
        min_ingles_nivel=params.ingles_min,
        top_n=params.top_n
    )

    inicializar_db(forzar=True)

    if resultado is None:
        return JSONResponse({
            "status": "error",
            "message": "Ningun candidato cumple con los criterios de filtrado."
        })

    top = resultado.to_dict(orient="records")
    return JSONResponse({"status": "success", "candidatos": top})


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    candidatos = obtener_candidatos()

    csv_exists = CANDIDATOS_CSV_PATH.exists()
    pdfs = sorted(RAG_DIR.glob("*.pdf"))
    pdf_exists = len(pdfs) > 0
    filtered_exists = TOP_CANDIDATOS_CSV_PATH.exists()

    csv_status = f'{STATUS_ICONS["uploaded"]} Cargado' if csv_exists else f'{STATUS_ICONS["missing"]} No cargado'
    pdf_status = f'{STATUS_ICONS["uploaded"]} {len(pdfs)} archivo(s)' if pdf_exists else f'{STATUS_ICONS["missing"]} Sin archivos'

    csv_filename = CANDIDATOS_CSV_PATH.name if csv_exists else "-"

    steps = [
        ("CSV", csv_exists, "Subir archivo CSV con datos de candidatos"),
        ("PDFs", pdf_exists, "Subir manuales corporativos para RAG"),
        ("Filtrar", filtered_exists, "Filtrar y seleccionar Top 5 candidatos"),
        ("Entrevistar", False, "Generar enlaces y realizar entrevistas"),
    ]

    steps_html = '<div style="display:flex;gap:1rem;justify-content:center;margin:1rem 0;">'
    for i, (label, done, _) in enumerate(steps):
        icon = "&#10003;" if done else str(i + 1)
        color = "var(--success)" if done else "var(--text-secondary)"
        border = f"2px solid {color}"
        bg = "rgba(16,185,129,0.1)" if done else "rgba(255,255,255,0.03)"
        steps_html += f"""
            <div style="display:flex;flex-direction:column;align-items:center;gap:0.3rem;">
                <div style="width:36px;height:36px;border-radius:50%;border:{border};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;color:{color};background:{bg};">
                    {icon}
                </div>
                <span style="font-size:0.7rem;color:var(--text-secondary);">{label}</span>
            </div>
        """
    steps_html += "</div>"

    config_card = f"""
    <div class="card" style="margin-bottom:2rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <h2 style="font-size:1.3rem;font-weight:700;">Configuracion del Sistema</h2>
        </div>

        {steps_html}

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1.5rem;">
            <div style="background:rgba(255,255,255,0.03);border:1px solid var(--border-color);border-radius:12px;padding:1.2rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                    <div>
                        <strong style="font-size:1rem;">CSV Candidatos</strong>
                        <div style="color:var(--text-secondary);font-size:0.8rem;margin-top:0.2rem;">{csv_filename}</div>
                    </div>
                    <div style="font-size:0.9rem;">{csv_status}</div>
                </div>
                <div style="display:flex;gap:0.5rem;">
                    <label class="btn" style="padding:0.4rem 0.8rem;font-size:0.85rem;cursor:pointer;">
                        Subir CSV
                        <input type="file" accept=".csv" onchange="subirCSV(this)" style="display:none;">
                    </label>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.03);border:1px solid var(--border-color);border-radius:12px;padding:1.2rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                    <div>
                        <strong style="font-size:1rem;">PDFs Base de Conocimiento</strong>
                        <div style="color:var(--text-secondary);font-size:0.8rem;margin-top:0.2rem;">{len(pdfs)} archivo(s)</div>
                    </div>
                    <div style="font-size:0.9rem;">{pdf_status}</div>
                </div>
                <div id="pdf-list" style="margin-bottom:0.8rem;">
                    {"".join(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;border-bottom:1px solid var(--border-color);font-size:0.85rem;"><span>{p.name}</span><button onclick="eliminarPDF(\'{p.name}\')" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:0.8rem;">&times;</button></div>' for p in pdfs)}
                </div>
                <div style="display:flex;gap:0.5rem;">
                    <label class="btn" style="padding:0.4rem 0.8rem;font-size:0.85rem;cursor:pointer;">
                        Subir PDF
                        <input type="file" accept=".pdf" onchange="subirPDF(this)" style="display:none;">
                    </label>
                </div>
            </div>
        </div>

        <div style="margin-top:1.5rem;border-top:1px solid var(--border-color);padding-top:1.2rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;cursor:pointer;" onclick="toggleFiltros()">
                <h3 style="font-size:1.1rem;font-weight:600;">Filtros de Busqueda</h3>
                <span id="filtros-toggle" style="color:var(--text-secondary);font-size:1.2rem;">&#9650;</span>
            </div>
            <div id="filtros-body" style="display:none;">

                <div style="margin-bottom:1rem;">
                    <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;color:var(--text-secondary);">Tecnologias (separadas por coma)</div>
                    <input type="text" id="filtro-tecnologias" value="" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                </div>

                <div style="margin-bottom:1rem;">
                    <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;color:var(--text-secondary);">Soft Skills (separadas por coma)</div>
                    <input type="text" id="filtro-soft_skills" value="" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                </div>

                <div style="margin-bottom:1rem;">
                    <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;color:var(--text-secondary);">Roles (separados por coma)</div>
                    <input type="text" id="filtro-roles" value="" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">

                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1rem;margin-bottom:1rem;">
                    <div>
                        <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;color:var(--text-secondary);">Remoto</div>
                        <select id="filtro-remoto" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                            <option value="true" selected>Requiere remoto</option>
                            <option value="false">No requiere remoto</option>
                        </select>
                    </div>
                    <div>
                        <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;color:var(--text-secondary);">Ingles minimo</div>
                        <select id="filtro-ingles" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                            <option value="B1">B1</option>
                            <option value="B2" selected>B2</option>
                            <option value="C1">C1</option>
                            <option value="C2">C2</option>
                        </select>
                    </div>
                    <div>
                        <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;color:var(--text-secondary);">Exp. minima (a\u00f1os)</div>
                        <input type="number" id="filtro-exp" value="2" min="0" max="20" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                    </div>
                    <div>
                        <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;color:var(--text-secondary);">Skill min.</div>
                        <input type="number" id="filtro-skill" value="5" min="1" max="10" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                    </div>
                    <div>
                        <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;color:var(--text-secondary);">Top N</div>
                        <input type="number" id="filtro-topn" value="5" min="1" max="20" style="width:100%;padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid var(--border-color);border-radius:6px;color:var(--text-primary);font-size:0.85rem;">
                    </div>
                </div>

            </div>
        </div>

        <div style="margin-top:1.5rem;text-align:center;">
            <button onclick="filtrarCandidatos()" class="btn" id="btn-filtrar" {"disabled" if not csv_exists else ""}
                    style="padding:0.6rem 2rem;font-size:1rem;{"opacity:0.5;cursor:not-allowed;" if not csv_exists else ""}">
                Filtrar Top Candidatos
            </button>
            <div id="filter-status" style="margin-top:0.5rem;color:var(--text-secondary);font-size:0.85rem;">
                {"Sube un archivo CSV para habilitar el filtrado." if not csv_exists else "Listo para filtrar." if not filtered_exists else "Filtrado completado. Los candidatos aparecen en la tabla de abajo."}
            </div>
        </div>
    </div>
    """

    # Construir filas de la tabla de candidatos
    rows_html = ""
    for cid, c in candidatos.items():
        estado = c["estado_entrevista"]
        if estado == "Pendiente":
            badge = f'<span class="badge badge-pending">{estado}</span>'
        elif estado == "Enlace Generado":
            badge = f'<span class="badge badge-generated">{estado}</span>'
        elif estado == "En Proceso":
            badge = f'<span class="badge badge-active">{estado}</span>'
        else:
            badge = f'<span class="badge badge-completed">{estado}</span>'

        link_entrevista = f"{request.base_url}entrevista/{cid}"

        btn_generar = f"""
            <button onclick="generarEnlace('{cid}')" class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;">
                Generar Enlace
            </button>
        """

        btn_copiar = f"""
            <button onclick="copiarEnlace('{link_entrevista}')" class="btn btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;">
                Copiar Link
            </button>
        """

        btn_reporte = f"""
            <a href="/reporte/{cid}" target="_blank" class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                Ver Reporte
            </a>
        """

        btn_reiniciar = f"""
            <button onclick="reiniciarEntrevista('{cid}')" class="btn btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.85rem; color: #f87171;">
                Reiniciar
            </button>
        """

        acciones = ""
        if estado == "Pendiente":
            acciones = btn_generar
        elif estado == "Enlace Generado":
            acciones = f'<div style="display:flex; gap:0.5rem;">{btn_copiar} {btn_reiniciar}</div>'
        elif estado == "En Proceso":
            acciones = f'<div style="display:flex; gap:0.5rem;">{btn_copiar} {btn_reiniciar}</div>'
        elif estado == "Completada":
            acciones = f'<div style="display:flex; gap:0.5rem;">{btn_reporte} {btn_reiniciar}</div>'

        rows_html += f"""
        <tr class="candidate-row">
            <td style="padding: 1.2rem; font-weight: 600;">{c['nombre']}</td>
            <td style="padding: 1.2rem; color: var(--text-secondary);">{c['anios_experiencia']} anos</td>
            <td style="padding: 1.2rem; text-align: center;"><span style="font-weight: 700; color: #a855f7;">{c['skill_score']}/10</span></td>
            <td style="padding: 1.2rem; text-align: center;">{badge}</td>
            <td style="padding: 1.2rem;">
                <div style="max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-secondary); font-size: 0.85rem;" title="{c['resumen_cv']}">
                    {c['resumen_cv']}
                </div>
            </td>
            <td style="padding: 1.2rem; text-align: right;">{acciones}</td>
        </tr>
        """

    if not rows_html and not filtered_exists:
        table_section = """
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;color:var(--text-secondary);">?</div>
            <h2 style="font-size:1.3rem;font-weight:700;margin-bottom:0.5rem;">No hay candidatos filtrados</h2>
            <p style="color:var(--text-secondary);">Sube un archivo CSV y haz clic en "Filtrar Top 5 Candidatos" para comenzar.</p>
        </div>
        """
    elif not rows_html:
        table_section = """
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;color:var(--warning);">!</div>
            <h2 style="font-size:1.3rem;font-weight:700;margin-bottom:0.5rem;">Sin candidatos disponibles</h2>
            <p style="color:var(--text-secondary);">No se encontraron candidatos en la base de datos. Vuelve a filtrar.</p>
        </div>
        """
    else:
        table_section = f"""
        <div class="card" style="overflow-x: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h2 style="font-size: 1.3rem; font-weight: 700;">Top Candidatos Evaluados</h2>
                <button onclick="recargarPagina()" class="btn btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;">
                    Actualizar Lista
                </button>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Candidato</th>
                        <th>Experiencia</th>
                        <th style="text-align: center;">Skill Score</th>
                        <th style="text-align: center;">Estado</th>
                        <th>Resumen CV</th>
                        <th style="text-align: right;">Acciones</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Panel de Seleccion - Reclutamiento Inteligente</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;800&display=swap" rel="stylesheet">
        <style>
            {ESTILOS_COMUNES}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1.5rem;
            }}
            th {{
                text-align: left;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.03);
                border-bottom: 1px solid var(--border-color);
                color: var(--text-secondary);
                font-weight: 600;
                font-size: 0.9rem;
            }}
            .candidate-row {{
                border-bottom: 1px solid var(--border-color);
                transition: background 0.3s ease;
            }}
            .candidate-row:hover {{
                background: rgba(255, 255, 255, 0.02);
            }}
            .hero {{
                margin-bottom: 2rem;
                text-align: center;
                padding: 2rem 0;
            }}
            .hero h1 {{
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                letter-spacing: -0.5px;
            }}
            .hero p {{
                color: var(--text-secondary);
                font-size: 1.1rem;
                max-width: 600px;
                margin: 0 auto;
            }}

            select option {{
                background: #1a1f2e;
                color: #f3f4f6;
            }}
            .toast {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #10b981;
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
                display: none;
                z-index: 100;
                animation: slideIn 0.3s ease;
            }}
            @keyframes slideIn {{
                from {{ transform: translateY(100%); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <header class="header">
            <div class="header-content">
                <div class="logo">Portal de Reclutamiento</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem;">
                    Reclutador Inteligente
                </div>
            </div>
        </header>

        <div class="container">
            <div class="hero">
                <h1>Panel de Reclutamiento</h1>
                <p>Carga los datos de candidatos y la base de conocimiento, filtra el Top 5, y gestiona las entrevistas virtuales.</p>
            </div>

            {config_card}

            {table_section}
        </div>

        <div id="toast" class="toast"></div>

        <script>
            function recargarPagina() {{
                location.reload();
            }}

            function mostrarToast(mensaje, esError = false) {{
                const toast = document.getElementById("toast");
                toast.innerText = mensaje;
                toast.style.background = esError ? "#ef4444" : "#10b981";
                toast.style.display = "block";
                setTimeout(() => {{
                    toast.style.display = "none";
                }}, 3000);
            }}

            async function subirCSV(input) {{
                const file = input.files[0];
                if (!file) return;
                const form = new FormData();
                form.append("file", file);
                try {{
                    const res = await fetch("/api/upload-csv", {{ method: "POST", body: form }});
                    if (res.ok) {{
                        mostrarToast("CSV subido correctamente.");
                        setTimeout(recargarPagina, 1000);
                    }} else {{
                        const err = await res.json();
                        mostrarToast(err.detail || "Error al subir CSV", true);
                    }}
                }} catch(e) {{
                    mostrarToast("Error de conexion", true);
                }}
            }}

            async function subirPDF(input) {{
                const file = input.files[0];
                if (!file) return;
                const form = new FormData();
                form.append("file", file);
                try {{
                    const res = await fetch("/api/upload-pdf", {{ method: "POST", body: form }});
                    if (res.ok) {{
                        mostrarToast("PDF subido y RAG configurado.");
                        setTimeout(recargarPagina, 1000);
                    }} else {{
                        const err = await res.json();
                        mostrarToast(err.detail || "Error al subir PDF", true);
                    }}
                }} catch(e) {{
                    mostrarToast("Error de conexion", true);
                }}
            }}

            async function eliminarPDF(filename) {{
                if (!confirm(`Eliminar ${{filename}}?`)) return;
                try {{
                    const res = await fetch(`/api/delete-pdf/${{filename}}`, {{ method: "DELETE" }});
                    if (res.ok) {{
                        mostrarToast("PDF eliminado.");
                        setTimeout(recargarPagina, 1000);
                    }} else {{
                        mostrarToast("Error al eliminar PDF", true);
                    }}
                }} catch(e) {{
                    mostrarToast("Error de conexion", true);
                }}
            }}

            function toggleFiltros() {{
                const body = document.getElementById("filtros-body");
                const toggle = document.getElementById("filtros-toggle");
                if (body.style.display === "none") {{
                    body.style.display = "block";
                    toggle.innerHTML = "&#9660;";
                }} else {{
                    body.style.display = "none";
                    toggle.innerHTML = "&#9650;";
                }}
            }}

            function obtenerDeInput(name) {{
                const val = document.getElementById("filtro-" + name)?.value || "";
                return val.split(",").map(s => s.trim()).filter(Boolean);
            }}

            async function filtrarCandidatos() {{
                const btn = document.getElementById("btn-filtrar");
                const status = document.getElementById("filter-status");
                btn.disabled = true;
                btn.style.opacity = "0.5";
                status.innerText = "Filtrando candidatos...";

                const payload = {{
                    tecnologias: obtenerDeInput("tecnologias"),
                    soft_skills: obtenerDeInput("soft_skills"),
                    roles: obtenerDeInput("roles"),
                    remoto: document.getElementById("filtro-remoto").value === "true",
                    ingles_min: document.getElementById("filtro-ingles").value,
                    experiencia_min: parseInt(document.getElementById("filtro-exp").value) || 2,
                    skill_min: parseInt(document.getElementById("filtro-skill").value) || 5,
                    top_n: parseInt(document.getElementById("filtro-topn").value) || 5
                }};

                try {{
                    const res = await fetch("/api/filtrar", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify(payload)
                    }});
                    const data = await res.json();
                    if (data.status === "success") {{
                        mostrarToast("Filtrado completado. " + data.candidatos.length + " candidatos encontrados.");
                        setTimeout(recargarPagina, 1000);
                    }} else {{
                        mostrarToast(data.message, true);
                        status.innerText = data.message;
                        btn.disabled = false;
                        btn.style.opacity = "1";
                    }}
                }} catch(e) {{
                    mostrarToast("Error de conexion", true);
                    status.innerText = "Error al filtrar.";
                    btn.disabled = false;
                    btn.style.opacity = "1";
                }}
            }}

            async function generarEnlace(cid) {{
                try {{
                    const res = await fetch(`/api/generar-enlace/${{cid}}`, {{ method: 'POST' }});
                    if (res.ok) {{
                        mostrarToast("Enlace generado con exito!");
                        setTimeout(recargarPagina, 1000);
                    }} else {{
                        mostrarToast("Error al generar enlace", true);
                    }}
                }} catch(e) {{
                    mostrarToast("Error de conexion", true);
                }}
            }}

            async function reiniciarEntrevista(cid) {{
                if (!confirm("Seguro de reiniciar esta entrevista? Se borrara el historial.")) return;
                try {{
                    const res = await fetch(`/api/reiniciar/${{cid}}`, {{ method: 'POST' }});
                    if (res.ok) {{
                        mostrarToast("Estado reiniciado.");
                        setTimeout(recargarPagina, 1000);
                    }} else {{
                        mostrarToast("Error al reiniciar", true);
                    }}
                }} catch(e) {{
                    mostrarToast("Error de conexion", true);
                }}
            }}

            function copiarEnlace(link) {{
                navigator.clipboard.writeText(link).then(() => {{
                    mostrarToast("Enlace copiado al portapapeles!");
                }}).catch(() => {{
                    mostrarToast("Error al copiar enlace", true);
                }});
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/entrevista/{candidato_id}", response_class=HTMLResponse)
def candidate_interview_page(candidato_id: str):
    candidato = obtener_candidato(candidato_id)
    if not candidato:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    # Si la entrevista ya terminó, mostrar pantalla final
    if candidato["estado_entrevista"] == "Completada":
        html_completada = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Entrevista Completada</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;800&display=swap" rel="stylesheet">
            <style>
                {ESTILOS_COMUNES}
                body {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="card" style="max-width: 500px; text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">🎉</div>
                <h1 style="font-size: 1.8rem; font-weight: 800; margin-bottom: 1rem;">¡Entrevista Finalizada!</h1>
                <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem;">
                    Muchas gracias, <strong>{candidato['nombre']}</strong>. Hemos completado las preguntas técnicas. 
                    Tus respuestas han sido enviadas al equipo de selección y evaluadas exitosamente.
                </p>
                <div style="border-top: 1px solid var(--border-color); padding-top: 1.5rem; color: var(--text-secondary); font-size: 0.85rem;">
                    Puedes cerrar esta pestaña de forma segura.
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_completada)

    # Actualizar estado a "En Proceso"
    if candidato["estado_entrevista"] != "En Proceso":
        actualizar_candidato(candidato_id, {"estado_entrevista": "En Proceso"})

    html_chat = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Entrevista Técnica Virtual - {candidato['nombre']}</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;800&display=swap" rel="stylesheet">
        <style>
            {ESTILOS_COMUNES}
            body {{
                height: 100vh;
                display: flex;
                flex-direction: column;
            }}
            .chat-container {{
                flex-grow: 1;
                max-width: 800px;
                width: 100%;
                margin: 2rem auto;
                padding: 0 1.5rem;
                display: flex;
                flex-direction: column;
                height: calc(100vh - 120px);
            }}
            .chat-box {{
                flex-grow: 1;
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
                margin-bottom: 1rem;
            }}
            .message {{
                max-width: 80%;
                padding: 1rem;
                border-radius: 12px;
                line-height: 1.5;
                font-size: 0.95rem;
            }}
            .message-recruiter {{
                align-self: flex-start;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border-color);
                border-top-left-radius: 2px;
            }}
            .message-candidate {{
                align-self: flex-end;
                background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
                color: white;
                border-top-right-radius: 2px;
            }}
            .input-box {{
                display: flex;
                gap: 0.75rem;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--border-color);
                padding: 0.5rem;
                border-radius: 12px;
            }}
            .input-box input {{
                flex-grow: 1;
                background: transparent;
                border: none;
                outline: none;
                color: var(--text-primary);
                padding: 0.75rem;
                font-size: 1rem;
            }}
            .typing-indicator {{
                align-self: flex-start;
                background: rgba(255, 255, 255, 0.05);
                padding: 0.8rem 1.2rem;
                border-radius: 12px;
                display: none;
                align-items: center;
                gap: 5px;
            }}
            .dot {{
                width: 6px;
                height: 6px;
                background: var(--text-secondary);
                border-radius: 50%;
                animation: wave 1.2s infinite ease-in-out;
            }}
            .dot:nth-child(2) {{ animation-delay: 0.2s; }}
            .dot:nth-child(3) {{ animation-delay: 0.4s; }}
            @keyframes wave {{
                0%, 60%, 100% {{ transform: translateY(0); }}
                30% {{ transform: translateY(-4px); }}
            }}
            .chat-status {{
                text-align: center;
                color: var(--text-secondary);
                font-size: 0.8rem;
                padding-bottom: 0.5rem;
            }}
        </style>
    </head>
    <body>
        <header class="header">
            <div class="header-content">
                <div class="logo">Entrevista Técnica</div>
                <div style="font-weight: 600; color: #a855f7;">Candidato: {candidato['nombre']}</div>
            </div>
        </header>
        
        <div class="chat-container">
            <div class="chat-status" id="chat-progress">
                Preguntas Realizadas: {candidato['numero_pregunta']} de 3
            </div>
            
            <div class="chat-box" id="chat-box">
                <!-- Los mensajes se insertarán aquí dinámicamente -->
            </div>
            
            <div class="typing-indicator" id="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>

            <div class="input-box" id="input-container">
                <input type="text" id="user-input" placeholder="Escribe tu respuesta aquí..." onkeypress="handleKeyPress(event)" autofocus>
                <button onclick="enviarMensaje()" class="btn" id="send-btn">
                    Enviar ➔
                </button>
            </div>
        </div>

        <script>
            const chatBox = document.getElementById("chat-box");
            const userInput = document.getElementById("user-input");
            const sendBtn = document.getElementById("send-btn");
            const typingIndicator = document.getElementById("typing-indicator");
            const chatProgress = document.getElementById("chat-progress");
            const inputContainer = document.getElementById("input-container");

            const candidatoId = "{candidato_id}";
            let historia = {json.dumps(candidato['historial_mensajes'])};
            let totalPreguntas = {candidato['numero_pregunta']};

            function agregarMensaje(rol, texto) {{
                const div = document.createElement("div");
                div.className = `message ${{rol === 'human' ? 'message-candidate' : 'message-recruiter'}}`;
                div.innerText = texto;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }}

            // Cargar historial previo
            if (historia.length > 0) {{
                historia.forEach(msg => {{
                    agregarMensaje(msg.role, msg.content);
                }});
            }} else {{
                // Iniciar la entrevista automáticamente
                iniciarEntrevista();
            }}

            async function iniciarEntrevista() {{
                toggleCargando(true);
                try {{
                    const res = await fetch(`/api/entrevista/${{candidatoId}}/mensaje`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ mensaje: "Hola, estoy listo para comenzar." }})
                    }});
                    const data = await res.json();
                    
                    if (data.respuesta) {{
                        agregarMensaje("ai", data.respuesta);
                        actualizarProgreso(data.numero_pregunta);
                    }}
                }} catch(e) {{
                    agregarMensaje("ai", "Error al conectar con el entrevistador. Por favor, recarga la página.");
                }} finally {{
                    toggleCargando(false);
                }}
            }}

            function toggleCargando(cargando) {{
                typingIndicator.style.display = cargando ? "flex" : "none";
                userInput.disabled = cargando;
                sendBtn.disabled = cargando;
            }}

            function actualizarProgreso(num) {{
                totalPreguntas = num;
                chatProgress.innerText = `Preguntas Realizadas: ${{num}} de 3`;
            }}

            function handleKeyPress(e) {{
                if (e.key === "Enter" && !userInput.disabled) {{
                    enviarMensaje();
                }}
            }}

            async function enviarMensaje() {{
                const texto = userInput.value.trim();
                if (!texto) return;

                agregarMensaje("human", texto);
                userInput.value = "";
                toggleCargando(true);

                try {{
                    const res = await fetch(`/api/entrevista/${{candidatoId}}/mensaje`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ mensaje: texto }})
                    }});
                    const data = await res.json();
                    
                    if (data.respuesta) {{
                        agregarMensaje("ai", data.respuesta);
                        actualizarProgreso(data.numero_pregunta);
                    }}
                    
                    if (data.finalizada) {{
                        mostrarFin();
                    }}
                }} catch(e) {{
                    agregarMensaje("ai", "Ocurrió un error. Intenta nuevamente.");
                }} finally {{
                    toggleCargando(false);
                }}
            }}

            function mostrarFin() {{
                inputContainer.innerHTML = `<div style="text-align: center; width: 100%; padding: 1rem; font-weight: 700; color: var(--success);">
                    Entrevista completada. Redireccionando...
                </div>`;
                setTimeout(() => {{
                    location.reload();
                }}, 2000);
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_chat)


@app.post("/api/entrevista/{candidato_id}/mensaje")
async def receive_candidate_message(candidato_id: str, request: MensajeRequest):
    global retriever_instancia
    candidato = obtener_candidato(candidato_id)
    if not candidato:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    if candidato["estado_entrevista"] == "Completada":
        return JSONResponse({
            "respuesta": "La entrevista ya ha sido completada.",
            "finalizada": True,
            "numero_pregunta": 3
        })

    mensaje_usuario = request.mensaje.strip()

    thread_id = f"sesion_{candidato_id}"
    graph = construir_grafo()

    # Recuperar el número de pregunta actual y el contexto
    numero_pregunta = candidato["numero_pregunta"]

    # RAG (opcional)
    contexto = ""
    if retriever_instancia:
        consulta_rag = f"Preguntas técnicas para {candidato['resumen_cv']}"
        documentos = retriever_instancia.invoke(consulta_rag)
        contexto = "\n\n".join([doc.page_content for doc in documentos])

    # Invocar el paso de LangGraph
    # Usamos HumanMessage
    estado = {
        "messages": [HumanMessage(content=mensaje_usuario)],
        "nombre": candidato["nombre"],
        "anios_experiencia": int(candidato["anios_experiencia"]),
        "resumen_cv": candidato["resumen_cv"],
        "contexto_documentos": contexto,
        "numero_pregunta": numero_pregunta
    }

    respuesta_state = graph.invoke(
        estado,
        config={"configurable": {"thread_id": thread_id}}
    )

    ultimo_mensaje = respuesta_state["messages"][-1].content
    nuevo_numero_pregunta = respuesta_state["numero_pregunta"]

    # Agregar mensajes a la base de datos
    historial = candidato["historial_mensajes"]
    # Registrar el mensaje enviado por el usuario (si no es el trigger inicial)
    if mensaje_usuario != "Hola, estoy listo para comenzar.":
        historial.append({"role": "human", "content": mensaje_usuario})
    # Registrar la respuesta del bot
    historial.append({"role": "ai", "content": ultimo_mensaje})

    # Verificar si finaliza la entrevista
    finalizada = nuevo_numero_pregunta >= 4 or mensaje_usuario.lower() == "salir"

    campos_actualizacion = {
        "historial_mensajes": historial,
        "numero_pregunta": nuevo_numero_pregunta,
        "estado_entrevista": "En Proceso"
    }

    reporte_archivo = None
    if finalizada:
        print(f"\nFinalizando entrevista web para {candidato['nombre']}...")
        campos_actualizacion["estado_entrevista"] = "Completada"

        # Recuperar todos los mensajes del checkpointer de LangGraph
        estado_final = graph.get_state(config={"configurable": {"thread_id": thread_id}})
        mensajes_completos = estado_final.values.get("messages", [])

        if mensajes_completos:
            transcripcion = construir_transcripcion(mensajes_completos)
            evaluacion = evaluar_entrevista(transcripcion)
            reporte_md = generar_reporte_markdown(candidato, evaluacion)

            reporte_archivo = f"{candidato['nombre']}.md"
            ruta_reporte = REPORTES_DIR / reporte_archivo
            with open(ruta_reporte, "w", encoding="utf-8") as f:
                f.write(reporte_md)
            campos_actualizacion["reporte_generado"] = reporte_archivo

    actualizar_candidato(candidato_id, campos_actualizacion)

    return JSONResponse({
        "respuesta": ultimo_mensaje,
        "finalizada": finalizada,
        "numero_pregunta": nuevo_numero_pregunta
    })


@app.post("/api/generar-enlace/{candidato_id}")
def api_generar_enlace(candidato_id: str):
    candidato = obtener_candidato(candidato_id)
    if not candidato:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    actualizar_candidato(candidato_id, {"estado_entrevista": "Enlace Generado"})
    return JSONResponse({"status": "success"})


@app.post("/api/reiniciar/{candidato_id}")
def api_reiniciar(candidato_id: str):
    candidato = obtener_candidato(candidato_id)
    if not candidato:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    # Limpiar el checkpointer de LangGraph para esta sesión recreando el grafo
    # Nota: LangGraph MemorySaver almacena estados en memoria. Para este thread_id,
    # simplemente podemos sobrescribir el estado o dejar que se cree un nuevo thread_id
    # al cambiar el id_candidato en config, o podemos cambiar el estado.
    # Para ser simples, el bot asume el número de pregunta como 0 y limpia su base de datos.
    actualizar_candidato(candidato_id, {
        "estado_entrevista": "Pendiente",
        "historial_mensajes": [],
        "numero_pregunta": 0,
        "reporte_generado": None
    })
    return JSONResponse({"status": "success"})


@app.get("/reporte/{candidato_id}", response_class=HTMLResponse)
def view_report(candidato_id: str):
    candidato = obtener_candidato(candidato_id)
    if not candidato or not candidato["reporte_generado"]:
        raise HTTPException(status_code=404, detail="Reporte no disponible para este candidato.")

    ruta_reporte = REPORTES_DIR / candidato["reporte_generado"]
    if not ruta_reporte.exists():
        raise HTTPException(status_code=404, detail="El archivo físico del reporte no fue encontrado.")

    with open(ruta_reporte, "r", encoding="utf-8") as f:
        contenido_md = f.read()

    # Convertir markdown a HTML
    contenido_html = markdown.markdown(contenido_md)

    html_reporte = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Evaluación - {candidato['nombre']}</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;800&display=swap" rel="stylesheet">
        <style>
            {ESTILOS_COMUNES}
            body {{
                padding: 2rem 1rem;
            }}
            .report-container {{
                max-width: 800px;
                margin: 0 auto;
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 3rem;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            }}
            h1, h2, h3, h4 {{
                font-family: 'Outfit', sans-serif;
                margin-bottom: 1rem;
                margin-top: 2rem;
                color: var(--text-primary);
            }}
            h1 {{
                font-size: 2.2rem;
                font-weight: 800;
                margin-top: 0;
                background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 1rem;
            }}
            h2 {{
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--accent-secondary);
            }}
            p, li {{
                font-size: 1.05rem;
                line-height: 1.6;
                color: var(--text-primary);
                margin-bottom: 1rem;
            }}
            ul {{
                margin-left: 2rem;
                margin-bottom: 1.5rem;
            }}
            hr {{
                border: 0;
                height: 1px;
                background: var(--border-color);
                margin: 2rem 0;
            }}
            .back-btn {{
                margin-bottom: 1.5rem;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <a href="/" class="btn btn-secondary back-btn">◀ Volver al Panel</a>
            <div class="markdown-body">
                {contenido_html}
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_reporte)
