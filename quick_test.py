#!/usr/bin/env python3
"""
Script de prueba rápida para demostrar la clasificación de imágenes
Crea imágenes de prueba simples si no existen
"""

import os
from pathlib import Path

def crear_imagenes_prueba():
    """
    Crea imágenes de prueba simples usando PIL
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow no está instalado. Instala con: pip install Pillow")
        return False
    
    # Crear directorio de prueba
    test_dir = Path("images_test")
    test_dir.mkdir(exist_ok=True)
    
    print(f"Creando imágenes de prueba en '{test_dir}'...")
    
    # Colores y textos para diferentes imágenes
    ejemplos = [
        {"color": (255, 100, 100), "texto": "Imagen Roja", "nombre": "test_rojo.png"},
        {"color": (100, 255, 100), "texto": "Imagen Verde", "nombre": "test_verde.png"},
        {"color": (100, 100, 255), "texto": "Imagen Azul", "nombre": "test_azul.png"},
        {"color": (255, 255, 100), "texto": "Imagen Amarilla", "nombre": "test_amarillo.png"},
    ]
    
    for ejemplo in ejemplos:
        # Crear imagen
        img = Image.new('RGB', (400, 300), color=ejemplo["color"])
        draw = ImageDraw.Draw(img)
        
        # Añadir texto
        try:
            # Intentar usar una fuente por defecto
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            # Si no encuentra la fuente, usar la por defecto
            font = ImageFont.load_default()
        
        # Dibujar texto centrado
        text = ejemplo["texto"]
        # Calcular posición del texto (aproximadamente centrado)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (400 - text_width) // 2
        y = (300 - text_height) // 2
        
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
        
        # Guardar imagen
        filepath = test_dir / ejemplo["nombre"]
        img.save(filepath)
        print(f"   Creada: {filepath}")
    
    print(f"\n{len(ejemplos)} imágenes de prueba creadas en '{test_dir}'")
    return True


def probar_clasificacion():
    """
    Prueba la clasificación de imágenes con las imágenes de prueba
    """
    print("\n" + "="*70)
    print("PRUEBA DE CLASIFICACIÓN DE IMÁGENES")
    print("="*70)
    
    try:
        from classify_images_with_ollama import OllamaImageClassifier
    except ImportError:
        print("\nNo se puede importar OllamaImageClassifier")
        print("   Asegúrate de que classify_images_with_ollama.py esté en el directorio")
        return False
    
    # Crear clasificador
    print("\n1⃣ Creando clasificador...")
    classifier = OllamaImageClassifier()
    
    # Verificar conexión
    print("\n2⃣ Verificando conexión con Ollama...")
    if not classifier.check_connection():
        print("\nNo se puede conectar con Ollama")
        print("\nPara ejecutar esta prueba:")
        print("   1. Inicia Ollama: ollama serve")
        print("   2. Instala un modelo: ollama pull gemma3:27b-it-qat")
        print("   3. Ejecuta este script nuevamente")
        return False
    
    # Verificar que existan imágenes
    test_dir = Path("images_test")
    if not test_dir.exists() or not list(test_dir.glob("*.png")):
        print(f"\nNo hay imágenes en '{test_dir}'")
        print("   Creando imágenes de prueba...")
        if not crear_imagenes_prueba():
            return False
    
    # Clasificar las imágenes
    print("\n3⃣ Clasificando imágenes de prueba...")
    print("   Este proceso puede tomar 1-2 minutos...")
    
    prompt = """Analiza esta imagen y describe:
1. Color predominante
2. Texto visible (si hay)
3. Características principales

Responde en JSON con: {"color": "...", "texto": "...", "descripcion": "..."}"""
    
    results = classifier.process_directory(
        directory_path=test_dir,
        prompt=prompt,
        output_file="test_classification_results.json"
    )
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("RESUMEN DE LA PRUEBA")
    print("="*70)
    
    if not results:
        print("No se obtuvieron resultados")
        return False
    
    successful = sum(1 for r in results if r.get('error') is None)
    print(f"\nTotal de imágenes: {len(results)}")
    print(f"Clasificaciones exitosas: {successful}")
    print(f"Errores: {len(results) - successful}")
    
    if successful > 0:
        print("\n¡PRUEBA EXITOSA!")
        print("Resultados guardados en 'test_classification_results.json'")
        print("\nEjemplo de clasificación:")
        for result in results[:1]:  # Mostrar solo el primero
            print(f"\n  Archivo: {result['file']}")
            classification = result['classification']
            if len(classification) > 200:
                classification = classification[:200] + "..."
            print(f"  Clasificación: {classification}")
    else:
        print("\nTodas las clasificaciones fallaron")
        print("Revisa los errores en los resultados")
    
    print("\n" + "="*70)
    return successful > 0


def main():
    """
    Función principal
    """
    print("="*70)
    print("PRUEBA RÁPIDA DEL SISTEMA DE CLASIFICACIÓN")
    print("="*70)
    
    print("\n¿Qué deseas hacer?")
    print("1. Solo crear imágenes de prueba")
    print("2. Crear imágenes Y ejecutar clasificación")
    print("3. Solo ejecutar clasificación (usa imágenes existentes)")
    print("0. Salir")
    
    try:
        opcion = input("\nSelecciona una opción (0-3): ").strip()
    except KeyboardInterrupt:
        print("\n\nCancelado")
        return
    
    if opcion == "0":
        print("\n¡Hasta luego!")
        return
    elif opcion == "1":
        crear_imagenes_prueba()
        print("\nPara clasificar estas imágenes, ejecuta:")
        print("   python quick_test.py")
        print("   y selecciona opción 3")
    elif opcion == "2":
        crear_imagenes_prueba()
        print("\nEsperando 2 segundos antes de clasificar...")
        import time
        time.sleep(2)
        probar_clasificacion()
    elif opcion == "3":
        probar_clasificacion()
    else:
        print("\nOpción no válida")


if __name__ == "__main__":
    main()
