#!/usr/bin/env python3
"""
Script para clasificar imágenes usando modelos de visión en Ollama
"""

import requests
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import gc
import os
import json
from typing import Dict, List, Optional, Union
from pathlib import Path

# Configuración por defecto
DEFAULT_MODEL = "gemma3:27b-it-qat"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaImageClassifier:
    """
    Clasificador de imágenes usando Ollama con modelos de visión
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL):
        """
        Inicializa el clasificador
        
        Args:
            model_name: Nombre del modelo de Ollama a usar
            ollama_url: URL del servidor Ollama
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.session = self._create_retry_session()
        
    def _create_retry_session(self, retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
        """Crea una sesión con capacidad de reintentos"""
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def check_connection(self) -> bool:
        """Verifica si Ollama está funcionando y el modelo está disponible"""
        try:
            print("Verificando conexión con Ollama...")
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                print(f"Conexión exitosa. Modelos disponibles: {model_names}")
                
                if self.model_name in model_names:
                    print(f"Modelo {self.model_name} encontrado")
                    return True
                else:
                    print(f"Advertencia: Modelo {self.model_name} no encontrado")
                    print("Modelos disponibles:")
                    for name in model_names:
                        print(f"  - {name}")
                    return False
            else:
                print(f"Error de conexión: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"No se pudo conectar con Ollama: {str(e)}")
            print("Sugerencias:")
            print(f"  - Verificar que Ollama esté ejecutándose: ollama serve")
            print(f"  - Verificar que el modelo esté instalado: ollama pull {self.model_name}")
            return False
    
    def download_image_as_base64(self, url: str) -> Optional[str]:
        """
        Descarga una imagen desde URL y la convierte a base64 JPEG
        
        Args:
            url: URL de la imagen
            
        Returns:
            String base64 de la imagen o None si falla
        """
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            with BytesIO(response.content) as buffer:
                img = Image.open(buffer).convert("RGB")
                output_buffer = BytesIO()
                img.save(output_buffer, format="JPEG")
                base64_str = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                
                # Limpiar memoria
                img.close()
                output_buffer.close()
                gc.collect()
                
                return base64_str
        except Exception as e:
            print(f"Error descargando imagen desde {url}: {str(e)}")
            return None
    
    def load_local_image_as_base64(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Carga una imagen local y la convierte a base64 JPEG
        
        Args:
            file_path: Ruta al archivo de imagen local
            
        Returns:
            String base64 de la imagen o None si falla
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Error: El archivo {file_path} no existe")
                return None
                
            img = Image.open(file_path).convert("RGB")
            output_buffer = BytesIO()
            img.save(output_buffer, format="JPEG")
            base64_str = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            # Limpiar memoria
            img.close()
            output_buffer.close()
            gc.collect()
            
            return base64_str
        except Exception as e:
            print(f"Error cargando imagen {file_path}: {str(e)}")
            return None
    
    def classify_image(self, base64_image: str, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Clasifica una imagen usando el modelo de Ollama
        
        Args:
            base64_image: Imagen codificada en base64
            prompt: Prompt de clasificación
            max_retries: Número máximo de reintentos
            
        Returns:
            Respuesta del modelo o None si falla
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                print(f"    Intento {attempt + 1}/{max_retries}...")
                response = self.session.post(
                    f"{self.ollama_url}/api/generate", 
                    json=payload,
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json()["response"].strip()
                    print(f"    Respuesta recibida exitosamente")
                    return result
                else:
                    print(f"    Error HTTP {response.status_code}: {response.text}")
            except Exception as e:
                print(f"    Error en intento {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"    Error después de {max_retries} intentos: {str(e)}")
                    return None
                print(f"    Esperando {2 ** attempt} segundos antes del siguiente intento...")
                time.sleep(2 ** attempt)
        
        return None
    
    def process_directory(self, directory_path: Union[str, Path], 
                         prompt: str,
                         output_file: str = "classification_results.json",
                         image_extensions: Optional[set] = None) -> List[Dict]:
        """
        Procesa todas las imágenes en un directorio
        
        Args:
            directory_path: Ruta al directorio con imágenes
            prompt: Prompt de clasificación a usar
            output_file: Nombre del archivo de salida JSON
            image_extensions: Extensiones de imagen a procesar (por defecto: jpg, jpeg, png, webp, bmp, gif)
            
        Returns:
            Lista de diccionarios con resultados
        """
        if image_extensions is None:
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            print(f"Error: El directorio '{directory_path}' no existe")
            return []
        
        # Obtener lista de archivos de imagen
        image_files = []
        for file in directory_path.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        if not image_files:
            print(f"Error: No se encontraron imágenes en '{directory_path}'")
            return []
        
        print(f"Se encontraron {len(image_files)} imágenes para procesar")
        print(f"Archivos: {[f.name for f in image_files]}")
        
        results = []
        
        # Procesar cada imagen
        for i, image_path in enumerate(image_files, 1):
            filename = image_path.name
            print(f"\n{'='*60}")
            print(f"PROCESANDO IMAGEN {i}/{len(image_files)}: {filename}")
            print('='*60)
            
            # Cargar imagen y convertir a base64
            base64_img = self.load_local_image_as_base64(image_path)
            
            if not base64_img:
                print(f"Error: No se pudo cargar la imagen {filename}")
                result = {
                    "file": filename,
                    "path": str(image_path),
                    "classification": "ERROR",
                    "error": "No se pudo cargar la imagen",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                continue
            
            print("Imagen cargada exitosamente")
            print("Clasificando imagen con modelo...")
            print("Este proceso puede tomar varios segundos...")
            
            # Clasificar imagen
            response = self.classify_image(base64_img, prompt)
            
            if response:
                print(f"\nRESULTADO:")
                print(f"{response[:200]}..." if len(response) > 200 else response)
            else:
                print(f"\nNo se pudo obtener respuesta del modelo")
            
            # Crear resultado
            result = {
                "file": filename,
                "path": str(image_path),
                "classification": response if response else "ERROR",
                "error": None if response else "No se pudo obtener respuesta del modelo",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)
        
        # Guardar resultados
        self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Guarda los resultados en un archivo JSON"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"\nResultados guardados en '{output_file}'")
        except Exception as e:
            print(f"\nError guardando resultados: {str(e)}")
    
    def classify_single_image(self, image_source: Union[str, Path], 
                            prompt: str,
                            is_url: bool = False) -> Optional[str]:
        """
        Clasifica una sola imagen (desde archivo local o URL)
        
        Args:
            image_source: Ruta al archivo o URL de la imagen
            prompt: Prompt de clasificación
            is_url: True si image_source es una URL, False si es un archivo local
            
        Returns:
            Respuesta del modelo o None si falla
        """
        print(f"\n{'='*60}")
        print(f"CLASIFICANDO IMAGEN: {image_source}")
        print('='*60)
        
        # Cargar imagen
        if is_url:
            print("Descargando imagen desde URL...")
            base64_img = self.download_image_as_base64(str(image_source))
        else:
            print("Cargando imagen desde archivo local...")
            base64_img = self.load_local_image_as_base64(image_source)
        
        if not base64_img:
            print("Error cargando imagen")
            return None
        
        print("Imagen cargada exitosamente")
        print("Clasificando imagen...")
        
        # Clasificar
        response = self.classify_image(base64_img, prompt)
        
        if response:
            print(f"\nRESULTADO:")
            print(response)
        else:
            print(f"\nNo se pudo obtener respuesta del modelo")
        
        return response


def create_example_prompt() -> str:
    """Crea un prompt para análisis de atractivo físico basado en OllamaAnalisis_CE.py"""
    return """You are a visual evaluator trained to assess human features in images with precision, objectivity, and fairness. Use a continuous scale from 0 to 100 to rate each criterion, not just discrete steps (like 20, 40, etc.). Return nuanced scores that reflect subtle differences. 
    
    If no person is clearly visible or evaluable, return "NA".

    Respond only with a number (0–100) or "NA". No additional commentary.

    Analyze the image and determine if a human body is clearly visible, considering possible image rotations.
    If a body is not clearly visible, return NA.

    Evaluate the body based on these criteria:
    - Proportional and balanced figure
    - Visible muscle tone (for males) or curves (for females)
    - Low body fat
    - Shows much of the body without clothing covering it (e.g., is in a bathing suit or underwear).
    - High waist-to-shoulder ratio (for men) or waist-to-hip ratio (for women)

    Rate the overall attractiveness on a continuous scale from 0 to 100, where:
    0 = very low, 100 = extremely high.
    You can use any integer value in between. Avoid rounding to common buckets (e.g., 20, 40, 60) unless justified by the image.

    Respond only with a number between 0 and 100, or "NA" if not evaluable. Do not include any other text.
    If the waist of the person is not clearly visible, return "NA"."""


def main():
    """Función principal de ejemplo"""
    print("="*70)
    print("CLASIFICADOR DE IMÁGENES CON OLLAMA")
    print("="*70)
    
    # Crear clasificador
    classifier = OllamaImageClassifier()
    
    # Verificar conexión
    if not classifier.check_connection():
        print("\nNo se puede continuar sin conexión con Ollama")
        print("\nPara iniciar Ollama:")
        print("   1. En una terminal: ollama serve")
        print(f"   2. Instalar el modelo: ollama pull {DEFAULT_MODEL}")
        return
    
    # Ejemplo de uso: procesar un directorio
    print("\n" + "="*70)
    print("EJEMPLO: Procesar directorio de imágenes")
    print("="*70)
    
    # Pedir ruta del directorio
    directory = input("\nIngresa la ruta del directorio con imágenes (o presiona Enter para usar 'images_test'): ").strip()
    if not directory:
        directory = "images_test"
    
    # Crear directorio de ejemplo si no existe
    if not Path(directory).exists():
        print(f"\nEl directorio '{directory}' no existe.")
        create_dir = input("¿Deseas crearlo? (s/n): ").strip().lower()
        if create_dir == 's':
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Directorio '{directory}' creado.")
            print(f"Coloca imágenes en este directorio y ejecuta el script nuevamente.")
        return
    
    # Usar prompt del repositorio si existe (fallback a create_example_prompt)
    prompt_path = Path("prompt_capital-erotico.txt")
    if prompt_path.exists():
        try:
            prompt = prompt_path.read_text(encoding='utf-8')
            print(f"Usando prompt de archivo: {prompt_path}")
        except Exception as e:
            print(f"No se pudo leer {prompt_path}: {e}. Usando prompt por defecto.")
            prompt = create_example_prompt()
    else:
        print("No se encontró 'prompt_capital-erotico.txt' en el repositorio. Usando prompt por defecto.")
        prompt = create_example_prompt()
    
    # Procesar directorio
    results = classifier.process_directory(
        directory_path=directory,
        prompt=prompt,
        output_file="image_classification_results.json"
    )
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"Total de imágenes procesadas: {len(results)}")
    successful = sum(1 for r in results if r.get('error') is None)
    print(f"Clasificaciones exitosas: {successful}")
    print(f"Errores: {len(results) - successful}")
    print("="*70)


if __name__ == "__main__":
    main()
