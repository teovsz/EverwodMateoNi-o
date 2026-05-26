# 🚀 EverwodMateoNi-o — Sistema Inteligente de Extracción y Agrupamiento de FAQs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-blue.svg)](https://www.postgresql.org/)
[![AI-Powered](https://img.shields.io/badge/AI-NLP%20%26%20Embeddings-success.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Microservices-orange.svg)]()

Este repositorio contiene el **Proyecto Final de Inteligencia Artificial**, desarrollado por **Mateo Niño**. Consiste en una plataforma automatizada y distribuida de microservicios para la ingesta de datos conversacionales históricos, procesamiento de lenguaje natural (NLP), agrupamiento semántico mediante embeddings y validación humana de Preguntas Frecuentes (FAQs).

El sistema está diseñado para resolver un problema crítico de negocio: transformar interacciones desestructuradas de soporte técnico/atención al cliente en una base de conocimiento optimizada y limpia de manera completamente autónoma.

## 🛠️ Arquitectura del Sistema y Componentes

El proyecto está modularizado en servicios especializados que interactúan de forma desacoplada para garantizar la escalabilidad:

### 📥 1. Pipeline de Ingesta (`ingest_service.py`)
Se conecta a la base de datos `everwod_db` y extrae cronológicamente los mensajes de la tabla `chat_messages`. Reconstruye el contexto del diálogo emparejando secuencialmente las consultas de los usuarios con las respuestas correspondientes emitidas por el asistente virtual.

### 🧠 2. Motor de Embeddings (`embed_service.py`)
Responsable de la vectorización de los textos. Transforma las cadenas de texto del usuario en vectores numéricos densos dentro de un espacio latente, permitiendo capturar el significado semántico profundo e intenciones detrás de cada pregunta, ignorando variaciones gramaticales o tipográficas puntuales.

### 📊 3. Agrupamiento y Generación (`suggestion_service.py`)
Utiliza algoritmos de clustering densos (**DBSCAN** parametrizado mediante variables de entorno) para aislar patrones de consultas repetitivas. Una vez detectado un clúster masivo de intenciones similares:
* Filtra duplicados y ruido sintáctico.
* Invoca un modelo de lenguaje de última generación local (`Qwen/Qwen2.5-0.5B-Instruct`) para consolidar las preguntas del grupo y redactar una respuesta canónica, formal y perfectamente estructurada para el FAQ sugerido.

### 🌐 4. Capa de Validación (`validation_service.py` & `dashboard.html`)
Expone una interfaz gráfica limpia y moderna a través de endpoints locales. Permite que los administradores del sistema examinen visualmente las propuestas generadas por la IA, aprueben las FAQs óptimas o descarten falsos positivos de manera intuitiva.

### ⏱️ 5. Orquestador Automático (`scheduler.py`)
Un servicio demonio en segundo plano encargado de despertar las tareas cronológicas cada 24 horas, automatizando la actualización de la base de conocimiento sin intervención humana continua.
