from pydantic import BaseModel, Field
from typing import List


class EvaluacionEntrevista(BaseModel):

    nivel_tecnico_detectado: str = Field(
        description="Nivel detectado: Junior, Mid o Senior"
    )

    score_tecnico: float = Field(
        description="Score técnico de 0 a 10"
    )

    habilidades_blandas: List[str] = Field(
        description="Soft skills detectadas"
    )

    fortalezas: List[str] = Field(
        description="Fortalezas técnicas detectadas"
    )

    debilidades: List[str] = Field(
        description="Debilidades detectadas"
    )

    tecnologias_mencionadas: List[str] = Field(
        description="Tecnologías mencionadas por el candidato"
    )

    recomendacion_contratacion: str = Field(
        description="Sí, No o Considerar"
    )

    resumen_evaluacion: str = Field(
        description="Resumen ejecutivo de la entrevista"
    )
