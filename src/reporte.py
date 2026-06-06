def generar_reporte_markdown(candidato, evaluacion):
    """Genera un reporte final estructurado en Markdown, alineado correctamente."""
    fortalezas_str = "\n".join(f"- {f}" for f in evaluacion.fortalezas)
    debilidades_str = "\n".join(f"- {d}" for d in evaluacion.debilidades)

    reporte = f"""# Reporte Final

## Información del candidato

- Nombre: {candidato['nombre']}
- Experiencia: {candidato['anios_experiencia']} años

---

## Evaluación técnica

- Nivel: {evaluacion.nivel_tecnico_detectado}
- Score: {evaluacion.score_tecnico}
- Recomendación: {evaluacion.recomendacion_contratacion}

---

## Fortalezas

{fortalezas_str}

---

## Debilidades

{debilidades_str}

---

## Resumen

{evaluacion.resumen_evaluacion}
"""
    return reporte
