import json
import pandas as pd
from src.config import TOP_CANDIDATOS_CSV_PATH, BASE_DIR

DB_PATH = BASE_DIR / "datos_entrevistas.json"


def inicializar_db(forzar=False):
    """Inicializa la base de datos JSON leyendo el CSV de top candidatos si no existe."""
    if not forzar and DB_PATH.exists():
        with open(DB_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass

    candidatos = {}
    if TOP_CANDIDATOS_CSV_PATH.exists():
        df = pd.read_csv(TOP_CANDIDATOS_CSV_PATH)
        # Tomar los primeros 5 candidatos
        for idx, row in df.head(5).iterrows():
            candidato_id = row.get("id_candidato", f"CAND_{idx+1:03d}")
            candidatos[candidato_id] = {
                "id_candidato": candidato_id,
                "nombre": row["nombre"],
                "anios_experiencia": int(row["anios_experiencia"]),
                "skill_score": float(row["skill_score"]),
                "remoto_preferencia": bool(row["remoto_preferencia"]),
                "resumen_cv": row["resumen_cv"],
                "score_final": float(row["score_final"]),
                "estado_entrevista": "Pendiente",  
                "historial_mensajes": [], 
                "numero_pregunta": 0,
                "reporte_generado": None
            }
        guardar_db(candidatos)
    else:
        print(f"Advertencia: No se encontró el archivo {TOP_CANDIDATOS_CSV_PATH} para inicializar la base de datos.")
    return candidatos


def guardar_db(candidatos):
    """Guarda la base de datos en un archivo JSON."""
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(candidatos, f, indent=4, ensure_ascii=False)


def obtener_candidatos():
    """Retorna la lista de todos los candidatos."""
    db = inicializar_db()
    return db


def obtener_candidato(candidato_id):
    """Retorna los datos de un candidato específico."""
    db = inicializar_db()
    return db.get(candidato_id)


def actualizar_candidato(candidato_id, campos_actualizados):
    """Actualiza campos específicos de un candidato."""
    db = inicializar_db()
    if candidato_id in db:
        db[candidato_id].update(campos_actualizados)
        guardar_db(db)
        return db[candidato_id]
    return None
