<div align="center">

# Clasificación Automática con LLMs

Clasificación automática de **textos** e **imágenes** mediante modelos de lenguaje grandes.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Características

| Capacidad | Detalle |
|-----------|---------|
| Clasificación de texto | OpenAI GPT-4o-mini |
| Clasificación de imágenes | Ollama con modelos de visión (local) |
| Procesamiento por lotes | Directorios completos de imágenes |
| Formatos de imagen | JPG, JPEG, PNG, WEBP, BMP, GIF |
| Evaluación | Métricas opcionales frente a etiquetas manuales |
| Verificación | Script de diagnóstico del entorno |

## Instalación

```bash
git clone https://github.com/tu-usuario/clasificacion-llms.git
cd clasificacion-llms
pip install -r requirements.txt
```

## Configuración

**Clasificación de texto (OpenAI)**

```bash
cp .env.example .env
# Edita .env y añade tu clave:
# OPENAI_API_KEY=tu-api-key-aqui
```

Obtén tu API key en https://platform.openai.com/api-keys

**Clasificación de imágenes (Ollama)**

```bash
# Instala Ollama desde https://ollama.ai
ollama serve                      # inicia el servidor
ollama pull gemma3:27b-it-qat     # instala un modelo de visión (otra terminal)
```

## Verificar instalación

```bash
python check_installation.py
```

Comprueba la versión de Python, las dependencias, la configuración de la API y el servidor de Ollama.

## Uso

### Clasificación de texto con GPT-4o-mini

El script `classify_with_gpt.py` clasifica frases en tres dimensiones (sense, reference, attribution).

```bash
# Clasificar tu propio dataset (CSV con una columna 'frase')
python classify_with_gpt.py --input mi_dataset.csv

# Clasificar y evaluar el desempeño frente a etiquetas manuales
python classify_with_gpt.py --input mi_dataset.csv --validate
```

Opciones principales:

| Flag | Descripción |
|------|-------------|
| `--input`, `-i` | CSV de entrada con una columna `frase`. |
| `--output`, `-o` | CSV de salida (por defecto `gpt_classification_results.csv`). |
| `--prompt`, `-p` | Archivo de prompt (por defecto `prompt_sistema-clasificacion-tridimensional.txt`). |
| `--validate` | Evalúa frente a las columnas `sense_manual`, `reference_manual`, `attribution_manual`. |

Las columnas `*_manual` son **opcionales**: solo se necesitan al usar `--validate`.

### Clasificación de imágenes con Ollama

**Uso interactivo**

```bash
python classify_images_with_ollama.py
```

**Uso programático**

```python
from classify_images_with_ollama import OllamaImageClassifier

classifier = OllamaImageClassifier(
    model_name="gemma3:27b-it-qat",
    ollama_url="http://localhost:11434"
)

if classifier.check_connection():
    # Una sola imagen local
    result = classifier.classify_single_image(
        image_source="path/to/image.jpg",
        prompt="Describe this image in detail.",
        is_url=False
    )
    print(result)

    # Un directorio completo
    results = classifier.process_directory(
        directory_path="images_folder",
        prompt="Describe this image in detail.",
        output_file="results.json"
    )
```

**Desde una URL**

```python
result = classifier.classify_single_image(
    image_source="https://example.com/image.jpg",
    prompt="Describe this image",
    is_url=True
)
```

## Resultados

**Clasificación de texto** — CSV con predicciones por dimensión y, con `--validate`,
accuracy y reporte de clasificación (precision, recall, F1) por dimensión.

**Clasificación de imágenes** — JSON con el resultado de cada imagen, resumen en
consola y manejo de errores por imagen.

```json
[
    {
        "file": "image1.jpg",
        "path": "/full/path/to/image1.jpg",
        "classification": "Detailed description...",
        "error": null,
        "timestamp": "2025-10-30 14:32:15"
    }
]
```

## Personalización de prompts

```python
object_prompt = """Identify all objects in this image.
For each object, provide: name, location, estimated size and color.
Respond in JSON format."""
```

Consulta [ADVANCED_PROMPTS.md](ADVANCED_PROMPTS.md) para más ejemplos especializados.

## Procesamiento por lotes

```python
from classify_images_with_ollama import OllamaImageClassifier

classifier = OllamaImageClassifier()
for dir_path in ["dir1", "dir2", "dir3"]:
    classifier.process_directory(
        directory_path=dir_path,
        prompt="Describe this image briefly",
        output_file=f"{dir_path}_results.json"
    )
```

## Solución de problemas

| Problema | Solución |
|----------|----------|
| Ollama no conecta | Ejecuta `ollama serve` y verifica con `ollama list` |
| Modelo no encontrado | Instala el modelo: `ollama pull gemma3:27b-it-qat` |
| Error de memoria | Las imágenes se optimizan automáticamente; reduce el tamaño si persiste |
| OpenAI timeout | Verifica conexión y API key. Hay 3 reintentos automáticos |
| Dependencias faltantes | Ejecuta `python check_installation.py` |

## Documentación adicional

- [QUICKSTART.md](QUICKSTART.md) — Guía de inicio rápido
- [ADVANCED_PROMPTS.md](ADVANCED_PROMPTS.md) — Ejemplos de prompts especializados
- [check_installation.py](check_installation.py) — Verificación del entorno
- [example_usage.py](example_usage.py) — Ejemplos interactivos

## Seguridad

- Nunca subas tu `.env` con API keys al repositorio.
- Usa `.env.example` como plantilla.
- Mantén las API keys en variables de entorno.
- Revisa el `.gitignore` antes de hacer commit.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## Agradecimientos

- [Ollama](https://ollama.ai) por los modelos de visión locales.
- [OpenAI](https://openai.com) por su API de clasificación de texto.
