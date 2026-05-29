#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias estén instaladas correctamente
"""

import sys
from pathlib import Path

print("=" * 70)
print("VERIFICACIÓN DE INSTALACIÓN")
print("=" * 70)

errors = []
warnings = []

# 1. Verificar versión de Python
print("\n1. Verificando versión de Python...")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 8:
    print(f"   [OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   [ERROR] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    errors.append("Python 3.8+ requerido")

# 2. Verificar dependencias para clasificación de texto
print("\n2. Verificando dependencias para clasificación de texto...")

try:
    import pandas
    print(f"   [OK] pandas {pandas.__version__}")
except ImportError:
    print("   [ERROR] pandas no instalado")
    errors.append("pip install pandas")

try:
    import openai
    print(f"   [OK] openai {openai.__version__}")
except ImportError:
    print("   [ERROR] openai no instalado")
    errors.append("pip install openai")

try:
    import sklearn
    print(f"   [OK] scikit-learn {sklearn.__version__}")
except ImportError:
    print("   [ERROR] scikit-learn no instalado")
    errors.append("pip install scikit-learn")

try:
    import numpy
    print(f"   [OK] numpy {numpy.__version__}")
except ImportError:
    print("   [ERROR] numpy no instalado")
    errors.append("pip install numpy")

# 3. Verificar dependencias para clasificación de imágenes
print("\n3. Verificando dependencias para clasificación de imágenes...")

try:
    import requests
    print(f"   [OK] requests {requests.__version__}")
except ImportError:
    print("   [ERROR] requests no instalado")
    errors.append("pip install requests")

try:
    from PIL import Image
    import PIL
    print(f"   [OK] Pillow {PIL.__version__}")
except ImportError:
    print("   [ERROR] Pillow no instalado")
    errors.append("pip install Pillow")

try:
    import tqdm
    print(f"   [OK] tqdm {tqdm.__version__}")
except ImportError:
    print("   [ERROR] tqdm no instalado")
    errors.append("pip install tqdm")

# 4. Verificar API key de OpenAI
print("\n4. Verificando configuración de OpenAI...")
import os
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"   [OK] OPENAI_API_KEY configurada (longitud: {len(api_key)} caracteres)")
else:
    print("   [AVISO] OPENAI_API_KEY no configurada")
    warnings.append("Configura: export OPENAI_API_KEY='tu-api-key'")

# 5. Verificar Ollama
print("\n5. Verificando instalación de Ollama...")
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"   [OK] Ollama está ejecutándose")
        print(f"   Modelos instalados: {len(models)}")
        for model in models:
            print(f"      - {model['name']}")
    else:
        print(f"   [ERROR] Ollama respondió con código {response.status_code}")
        warnings.append("Verifica la instalación de Ollama")
except Exception as e:
    print(f"   [AVISO] No se pudo conectar con Ollama")
    warnings.append("Inicia Ollama: ollama serve")

# 6. Verificar archivos del proyecto
print("\n6. Verificando archivos del proyecto...")

required_files = [
    "classify_with_gpt.py",
    "classify_images_with_ollama.py",
    "example_usage.py",
    "prompt_sistema-clasificacion-tridimensional.txt",
    "prompt_capital-erotico.txt",
    "requirements.txt",
    "README.md"
]

for file in required_files:
    if Path(file).exists():
        print(f"   [OK] {file}")
    else:
        print(f"   [ERROR] {file} no encontrado")
        errors.append(f"Archivo faltante: {file}")

# Resumen
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

if not errors and not warnings:
    print("\nTodo está configurado correctamente.")
    print("\nPróximos pasos:")
    print("   1. Para clasificación de imágenes: python classify_images_with_ollama.py")
    print("   2. Para clasificación de texto: python classify_with_gpt.py --input mi_dataset.csv")
    print("   3. Para ver ejemplos: python example_usage.py")
elif errors:
    print(f"\nSe encontraron {len(errors)} errores:")
    for error in errors:
        print(f"   - {error}")
    if warnings:
        print(f"\nTambién hay {len(warnings)} advertencias:")
        for warning in warnings:
            print(f"   - {warning}")
else:
    print(f"\nSe encontraron {len(warnings)} advertencias:")
    for warning in warnings:
        print(f"   - {warning}")
    print("\nEl sistema funciona pero algunas características pueden no estar disponibles.")

print("\n" + "=" * 70)

# Mostrar comando de instalación si hay errores
if errors:
    pip_commands = [e for e in errors if e.startswith("pip install")]
    if pip_commands:
        print("\nInstala todas las dependencias con:")
        print("   pip install -r requirements.txt")
        print("\n   O individualmente:")
        for cmd in pip_commands:
            print(f"   {cmd}")

sys.exit(1 if errors else 0)
