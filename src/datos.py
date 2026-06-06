import pandas as pd
import random
from src.config import CANDIDATOS_CSV_PATH

tecnologias = [
    "Django", "FastAPI", "Flask", "Express.js", "NestJS",
    "React", "Next.js", "Angular", "Vue", "Svelte",
    "Python", "JavaScript", "Java", "TypeScript", "C#", "Go", "Rust", "Kotlin", "Swift",
    "MySQL", "PostgreSQL", "MongoDB", "SQL Server", "Redis",
    "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Kafka",
]

soft_skills = [
    "trabajo en equipo",
    "comunicación efectiva",
    "liderazgo técnico",
    "adaptabilidad",
    "resolución de problemas",
    "proactividad",
    "pensamiento crítico",
    "creatividad",
    "colaboración",
    "gestión del tiempo",
    "negociación",
    "empatía",
]

roles = [
    "Backend Developer",
    "Software Engineer",
    "Python Developer",
    "Fullstack Developer",
    "Backend Engineer",
    "Machine Learning Engineer",
    "Frontend Developer",
    "Data Engineer",
    "Data Scientist",
    "DevOps Engineer",
    "QA Engineer",
    "Security Engineer",
    "Solutions Architect",
    "Technical Lead",
    "Site Reliability Engineer",
    "Mobile Developer",
    "Product Manager",
]

areas = [
    "automatización empresarial",
    "microservicios",
    "plataformas SaaS",
    "inteligencia artificial",
    "finanzas",
    "e-commerce",
    "educación",
    "salud",
    "tecnología financiera",
    "logística",
    "retail",
    "marketing digital",
    "ciber seguridad",
    "big data",
    "internet de las cosas",
    "realidad aumentada",
    "blockchain",
]

templates_resumen = [
    # Backend / API
    "Desarrollador {rol} con {exp} años de experiencia en {area}. Stack principal: {tech1}, {tech2}, Python. Destaca por {skill}.",
    "Backend engineer con {skill} especializado en APIs REST y microservicios. Experiencia con {tech1}, {tech2} y bases de datos SQL/NoSQL.",
    "Ingeniero backend con {exp} años construyendo sistemas escalables en {area}. Manejo avanzado de {tech1} y {tech2}.",
    "Backend developer con experiencia en integración de servicios, colas de mensajería y despliegues cloud usando {tech1}, {tech2} y Docker.",

    # Fullstack
    "Fullstack developer con {skill} y {exp} años construyendo soluciones end-to-end con {tech1} en frontend y {tech2} en backend.",
    "Fullstack engineer apasionado por {area}. Trabaja con {tech1}, {tech2} y bases de datos. Reconocido por su {skill}.",
    "Desarrollador fullstack con {skill}. Experiencia en frontend moderno ({tech1}), backend escalable ({tech2}) y cloud.",

    # Frontend
    "Frontend developer con {skill} especializado en {tech1} y ecosistema {tech2}. Crea interfaces modernas, accesibles y con buen rendimiento.",
    "Frontend engineer apasionado por UX/UI con {exp} años usando {tech1}, {tech2} y diseño responsive. Experiencia en {area}.",
    "Desarrollador frontend con {skill}. Domina {tech1}, {tech2} y testing de componentes. Enfoque en accesibilidad y rendimiento.",

    # Mobile
    "Mobile developer con {exp} años creando apps nativas y multiplataforma con {tech1} y {tech2}. Orientado a {area}.",
    "Desarrollador mobile con {skill} y experiencia en publicación de apps. Stack: {tech1}, {tech2}, CI/CD móvil.",

    # ML / Data
    "Machine Learning engineer con {exp} años construyendo pipelines de datos y modelos predictivos para {area}. Usa {tech1}, {tech2} y Python.",
    "Data scientist con {skill}. Experiencia en {area} análisis de datos, feature engineering y despliegue de modelos con {tech1} y {tech2}.",
    "Data engineer especializado en pipelines ETL, data lakes y procesamiento en tiempo real con {tech1}, {tech2} y Spark.",

    # DevOps / Cloud
    "DevOps engineer con {skill} y {exp} años automatizando infraestructura. Stack: {tech1}, {tech2}, Docker, Kubernetes y CI/CD.",
    "Site Reliability Engineer con experiencia en {area}. Garantiza disponibilidad y rendimiento usando {tech1}, {tech2} y monitoreo avanzado.",
    "Cloud engineer certificado en AWS/Azure. Diseña arquitecturas escalables con {tech1}, {tech2} y Terraform.",

    # QA / Testing
    "QA engineer con {skill} y {exp} años en automatización de pruebas. Experiencia con {tech1}, {tech2} y frameworks de testing.",
    "Ingeniero de calidad con enfoque en {area}. Implementa pruebas unitarias, de integración y E2E con {tech1} y {tech2}.",

    # Security
    "Security engineer con {skill}. Experiencia en ciberseguridad, análisis de vulnerabilidades y DevSecOps con {tech1} y {tech2}.",
    "Especialista en seguridad informática para {area}. Implementa políticas de seguridad, pentesting y monitoreo con {tech1}.",

    # Roles generales / liderazgo
    "Ingeniero de software con {skill} y {exp} años en {area}. Maneja {tech1}, {tech2} y metodologías ágiles. Mentor de equipos técnicos.",
    "Software engineer con experiencia en sistemas distribuidos y alta disponibilidad usando {tech1}, {tech2} y prácticas DevOps.",
    "Desarrollador polivalente con {skill}. Experiencia en todo el ciclo de vida del software con {tech1}, {tech2} y entrega continua.",
    "Technical lead con {exp} años liderando equipos en {area}. Stack técnico: {tech1}, {tech2}. Promueve {skill} en el equipo.",

    "Soy {rol} con {exp} años de experiencia en {area}. Stack principal: {tech1}, {tech2}. Destaco por mi {skill}.",
    "Mi perfil como {rol} combina {exp} años de experiencia, {tech1}, {tech2} y un fuerte enfoque en {area}.",
    "A lo largo de mi carrera como {rol} he trabajado en proyectos de {area}, utilizando tecnologías como {tech1}, {tech2} y destacando por mi {skill}.",
    "Como {rol} especializado en {area}, aporto {exp} años de experiencia, {tech1}, {tech2} y {skill}.",
    "Especializado en {area}, mi experiencia como {rol} abarca {tech1}, {tech2} y un sólido historial de {skill}.",
    "Como {rol}, combino {exp} años de experiencia con {tech1}, {tech2} para ofrecer soluciones innovadoras en {area}.",
    "Mi enfoque como {rol} se centra en {area}, con experiencia en {tech1}, {tech2} y un fuerte compromiso con la calidad y {skill}.",
    "Como {rol}, he desarrollado una sólida experiencia en {area}, trabajando con {tech1}, {tech2} y destacando por mi {skill}.",
]

nombres_ficticios = [
    "Alejandro", "María", "Carlos", "Ana", "Luis",
    "Laura", "Juan", "Sofía", "Pedro", "Elena",
    "Camila", "Andrés", "Valentina", "Julián",
    "Mateo", "Daniela", "Sebastián", "Natalia",
    "Miguel", "Paula", "Felipe", "Sara",
    "Tomás", "Lucía", "David", "Isabella",
    "Gabriel", "Manuela", "Samuel", "Mariana",
    "Nicolás", "Valeria", "Emilio", "Tatiana",
    "Kevin", "Catalina", "Ricardo", "Fernanda",
    "Diego", "Julia", "Martín", "Rocío",
    "Pablo", "Adriana", "Hugo", "Lorena",
    "Javier", "Claudia", "Iván", "Verónica",
]

apellidos_ficticios = [
    "Gómez", "Rodríguez", "López", "Martínez",
    "García", "Pérez", "Sánchez", "Fernández",
    "Ramírez", "Torres", "Vargas", "Morales",
    "Castro", "Rojas", "Navarro", "Ortega",
    "Silva", "Mendoza", "Herrera", "Jiménez",
    "Ruiz", "Medina", "Cruz", "Delgado",
    "Reyes", "Guerrero", "Campos", "Vega",
    "Fuentes", "Acosta", "Peña", "Cortés",
    "Flores", "Rivas", "Molina", "Cabrera",
    "Santos", "Aguilar", "Iglesias", "Paredes",
]

niveles_ingles = ["A1", "A2", "B1", "B2", "C1", "C2"]
pesos_ingles = [5, 10, 25, 30, 20, 10]


def generar_dataset_candidatos(num_candidatos=1000):
    """Genera un DataFrame con datos de candidatos ficticios."""
    candidatos = []

    for i in range(1, num_candidatos + 1):
        nombre_completo = (
            f"{random.choice(nombres_ficticios)} "
            f"{random.choice(apellidos_ficticios)}"
        )

        # Experiencia: Mantenemos la distribución sesgada, pero sumamos un decimal
        base_exp = random.choices(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            weights=[5, 10, 5, 3, 2, 1, 1, 0.5, 0.5]
        )[0]
        # Sumamos entre 0.0 y 0.9 años y redondeamos a 1 decimal (ej. 1.5, 3.2)
        anios_exp = round(base_exp + random.uniform(0, 0.9), 1)

        # Skill score: Correlacionado con la experiencia + ruido en formato decimal
        ruido_skill = random.uniform(-2.0, 3.0) 
        base_score = anios_exp + ruido_skill
        # Limitamos entre 1.0 y 10.0 y redondeamos a 1 decimal
        skill_sc = round(max(1.0, min(10.0, base_score)), 1)

        # Tecnologías: entre 2 y 5
        num_techs = random.randint(2, 5)
        techs = random.sample(tecnologias, num_techs)

        # Inglés con distribución realista
        nivel_ingles = random.choices(niveles_ingles, weights=pesos_ingles)[0]

        # Preferencia remota
        prefiere_remoto = random.choices(
            [True, False],
            weights=[65, 35]
        )[0]

        # Generar resumen con reemplazo de placeholders
        template = random.choice(templates_resumen)

        # Algunos templates usan {tech2}, otros no. Manejar faltantes.
        placeholders = {
            "rol": random.choice(roles),
            "exp": anios_exp, # Al inyectarse aquí, se verá como "1.5" en el texto
            "area": random.choice(areas),
            "tech1": techs[0],
            "tech2": techs[1] if len(techs) > 1 else techs[0],
            "skill": random.choice(soft_skills),
        }

        try:
            resumen = template.format(**placeholders)
        except KeyError:
            resumen = template.format(
                rol=placeholders["rol"],
                exp=placeholders["exp"],
                area=placeholders["area"],
                tech1=placeholders["tech1"],
                tech2=placeholders["tech2"],
                skill=placeholders["skill"],
            )

        candidatos.append({
            "id_candidato": f"CAND_{i:03d}",
            "nombre": nombre_completo,
            "anios_experiencia": anios_exp,
            "skill_score": skill_sc,
            "ingles_nivel": nivel_ingles,
            "remoto_preferencia": prefiere_remoto,
            "resumen_cv": resumen
        })

    return pd.DataFrame(candidatos)

def ejecutar_generacion(ruta_salida=CANDIDATOS_CSV_PATH, num_candidatos=1000):
    """Ejecuta la generación y la guarda en un CSV."""
    print("Generando candidatos ficticios...")
    df_candidatos = generar_dataset_candidatos(num_candidatos)
    df_candidatos.to_csv(ruta_salida, index=False)
    print(f"Dataset de {num_candidatos} candidatos generado en: {ruta_salida}")
    print(df_candidatos.head(5))
    return df_candidatos


if __name__ == "__main__":
    ejecutar_generacion()
