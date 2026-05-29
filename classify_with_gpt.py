#!/usr/bin/env python3
"""
Script para clasificar frases usando GPT-4o-mini via API de OpenAI.

Permite, opcionalmente, evaluar el desempeño comparando las predicciones con
etiquetas manuales (columnas *_manual del dataset de entrada).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report

# Configuración de la API
API_KEY = os.getenv('OPENAI_API_KEY')  # Usar variable de entorno para seguridad
MODEL = 'gpt-4o-mini-2024-07-18'  # Usamos el modelo disponible más cercano

# Nombre por defecto del archivo de prompt del repositorio
DEFAULT_PROMPT_FILE = 'prompt_sistema-clasificacion-tridimensional.txt'

# Cliente OpenAI (se inicializa de forma diferida en get_client)
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Devuelve un cliente de OpenAI, validando que exista la API key."""
    global _client
    if not API_KEY:
        raise ValueError(
            "Configura la variable de entorno OPENAI_API_KEY con tu clave de "
            "API de OpenAI."
        )
    if _client is None:
        _client = OpenAI(api_key=API_KEY)
    return _client


def load_prompt(prompt_file: str = DEFAULT_PROMPT_FILE) -> str:
    """Carga el prompt desde el archivo indicado."""
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de prompt '{prompt_file}'."
        )
    return path.read_text(encoding='utf-8')


def classify_sentence_with_gpt(sentence: str, prompt: str, max_retries: int = 3) -> Dict:
    """
    Clasifica una frase usando GPT-4o-mini
    """
    full_prompt = f"{prompt}\n\nClassify the following sentence:\n\"{sentence}\""

    for attempt in range(max_retries):
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a multilingual identity statement classifier. Always respond with valid JSON following the specified format."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,  # Baja temperatura para resultados más consistentes
                max_tokens=1500
            )

            response_text = response.choices[0].message.content.strip()

            # Intentar parsear JSON
            try:
                # Limpiar el texto si contiene marcas de código markdown
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                elif response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                result = json.loads(response_text)
                return result

            except json.JSONDecodeError:
                print(f"Error parsing JSON on attempt {attempt + 1} for sentence: {sentence[:50]}...")
                if attempt == max_retries - 1:
                    return create_error_response(sentence, "JSON parsing error")

        except Exception as e:
            print(f"API error on attempt {attempt + 1} for sentence: {sentence[:50]}... Error: {e}")
            if attempt == max_retries - 1:
                return create_error_response(sentence, f"API error: {e}")
            time.sleep(2)  # Esperar antes de reintentar

    return create_error_response(sentence, "Max retries exceeded")


def create_error_response(sentence: str, error: str) -> Dict:
    """Crea una respuesta de error en el formato esperado"""
    return {
        "sentences": [{
            "text": sentence,
            "sense": "ERROR",
            "reference": "ERROR",
            "attribution": "ERROR",
            "sense_justification": f"Error: {error}",
            "reference_justification": f"Error: {error}",
            "attribution_justification": f"Error: {error}"
        }],
        "summary": {
            "sense": "ERROR",
            "reference": "ERROR",
            "attribution": "ERROR",
            "sense_justification": f"Error: {error}",
            "reference_justification": f"Error: {error}",
            "attribution_justification": f"Error: {error}"
        }
    }


def extract_classification(gpt_response: Dict, sentence: str) -> Tuple[str, str, str]:
    """
    Extrae las clasificaciones de la respuesta de GPT
    """
    try:
        if "sentences" in gpt_response and len(gpt_response["sentences"]) > 0:
            sentence_data = gpt_response["sentences"][0]
            sense = sentence_data.get("sense", "ERROR")
            reference = sentence_data.get("reference", "ERROR")
            attribution = sentence_data.get("attribution", "ERROR")
            return sense, reference, attribution
        else:
            return "ERROR", "ERROR", "ERROR"
    except Exception as e:
        print(f"Error extracting classification for sentence: {sentence[:50]}... Error: {e}")
        return "ERROR", "ERROR", "ERROR"


def normalize_categories(category: str, dimension: str) -> str:
    """
    Normaliza las categorías para la comparación
    """
    if pd.isna(category) or category == "" or category == "ERROR":
        return "NA"

    category = str(category).strip()

    # Mapeo de categorías para Reference (jerarquía superior)
    if dimension == "reference":
        reference_mapping = {
            # Sin anclaje
            "Biosocial": "Sin anclaje", "Generic": "Sin anclaje",
            "Name": "Sin anclaje", "Gender": "Sin anclaje", "Age": "Sin anclaje",
            "Physical Characteristics": "Sin anclaje", "Health identity": "Sin anclaje",
            "Universal definition": "Sin anclaje", "Material partitive": "Sin anclaje",
            "Social partitive": "Sin anclaje",

            # Anclaje
            "Familiar": "Anclaje", "Groupal": "Anclaje", "Active": "Anclaje", "Social": "Anclaje",
            "Matrimonial": "Anclaje", "Partner": "Anclaje", "Nuclear family": "Anclaje",
            "Extended family": "Anclaje", "Home": "Anclaje", "Housing": "Anclaje",
            "Primary group": "Anclaje", "Secondary group": "Anclaje", "Generalized other": "Anclaje",
            "Job": "Anclaje", "Work role": "Anclaje", "Unemployment": "Anclaje",
            "Educational role": "Anclaje", "Complementary activity": "Anclaje",
            "Social class": "Anclaje", "Local": "Anclaje", "Local identity": "Anclaje",
            "Intermediate identity": "Anclaje", "State identity": "Anclaje",
            "Supranational identity": "Anclaje", "Marginal identity": "Anclaje",
            "Queer identity": "Anclaje", "Political identity": "Anclaje",
            "Sexual Orientation": "Anclaje", "Ethnic identity": "Anclaje",
            "Famous personalities": "Anclaje", "Religious identity": "Anclaje",
            "Linguistic reference": "Anclaje"
        }
        return reference_mapping.get(category, category)

    # Mapeo de categorías para Sense (jerarquía superior)
    elif dimension == "sense":
        sense_mapping = {
            # Consensual
            "Physical": "Consensual", "Collective": "Consensual", "Activity": "Consensual",
            "Property": "Consensual", "Narrative": "Consensual", "Global": "Consensual",

            # Subconsensual
            "Attitudinal": "Subconsensual", "Self-esteem": "Subconsensual",
            "Preference": "Subconsensual", "Beliefs": "Subconsensual",
            "Aspirations": "Subconsensual", "Self-doubt": "Subconsensual",
            "Nihilistic": "Subconsensual", "About others": "Subconsensual",
            "Test evasion": "Subconsensual", "Metaphor": "Subconsensual"
        }
        return sense_mapping.get(category, category)

    return category


def classify_dataset(
    df: pd.DataFrame,
    prompt: str,
    validate: bool = False,
    output_csv: str = "gpt_classification_results.csv",
    save_progress: bool = True,
) -> pd.DataFrame:
    """
    Clasifica todas las frases de un DataFrame y, opcionalmente, evalúa el
    desempeño contra las etiquetas manuales (*_manual).

    Args:
        df: DataFrame con al menos la columna 'frase'.
        prompt: Prompt de clasificación.
        validate: Si True, compara con las columnas sense_manual,
            reference_manual y attribution_manual y muestra métricas.
        output_csv: Archivo CSV de salida.
        save_progress: Si True, guarda progreso parcial cada 10 frases.

    Returns:
        DataFrame con los resultados de clasificación.
    """
    if 'frase' not in df.columns:
        raise ValueError("El dataset debe contener una columna 'frase'.")

    manual_cols = ['sense_manual', 'reference_manual', 'attribution_manual']
    if validate:
        missing = [c for c in manual_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Para validar, el dataset debe contener las columnas: "
                f"{', '.join(missing)}."
            )

    results = []
    print(f"Total de frases a clasificar: {len(df)}")
    print("\nIniciando clasificación...")
    start_time = time.time()

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row['frase']
        print(f"Procesando frase {position}/{len(df)}: {sentence[:50]}...")

        # Clasificar con GPT
        gpt_response = classify_sentence_with_gpt(sentence, prompt)

        # Extraer clasificaciones
        sense_pred, reference_pred, attribution_pred = extract_classification(gpt_response, sentence)

        result = {
            'bio_num': row.get('bio_num'),
            'frase_num': row.get('frase_num'),
            'frase': sentence,
            'sense_predicted': sense_pred,
            'reference_predicted': reference_pred,
            'attribution_predicted': attribution_pred,
            'gpt_response': json.dumps(gpt_response)
        }

        if validate:
            # Normalizar categorías verdaderas (de específicas a jerarquía superior)
            result['sense_true'] = normalize_categories(row['sense_manual'], 'sense')
            result['reference_true'] = (
                normalize_categories(row['reference_manual'], 'reference')
                if pd.notna(row['reference_manual']) else "NA"
            )
            result['attribution_true'] = (
                str(row['attribution_manual']) if pd.notna(row['attribution_manual']) else "NA"
            )

        results.append(result)

        # Pausa para evitar límites de rate
        time.sleep(0.5)

        # Guardar progreso cada 10 frases
        if save_progress and position % 10 == 0:
            pd.DataFrame(results).to_csv(f"temp_results_{position}.csv", index=False)
            print(f"Progreso guardado: {position} frases procesadas")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    elapsed_time = time.time() - start_time
    print(f"\nClasificación completada en {elapsed_time:.2f} segundos")
    print("\nArchivos generados:")
    print(f"- {output_csv}: Resultados completos de clasificación")

    if validate:
        evaluate_results(df_results)

    return df_results


def evaluate_results(df_results: pd.DataFrame) -> None:
    """
    Calcula y muestra métricas de evaluación por dimensión, comparando las
    columnas *_true y *_predicted (ignorando filas con valor ERROR).
    """
    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL DESEMPEÑO")
    print("=" * 60)

    for dimension in ("sense", "reference", "attribution"):
        true_col = f"{dimension}_true"
        pred_col = f"{dimension}_predicted"
        if true_col not in df_results.columns or pred_col not in df_results.columns:
            continue

        mask = (df_results[pred_col] != "ERROR") & (df_results[true_col] != "ERROR")
        y_true = df_results.loc[mask, true_col].astype(str)
        y_pred = df_results.loc[mask, pred_col].astype(str)

        print(f"\nDimensión: {dimension}")
        if len(y_true) == 0:
            print("  Sin datos válidos para evaluar.")
            continue

        accuracy = accuracy_score(y_true, y_pred)
        print(f"  Accuracy: {accuracy:.3f} ({len(y_true)} frases evaluadas)")
        print(classification_report(y_true, y_pred, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica frases con GPT-4o-mini y, opcionalmente, evalúa el desempeño."
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="CSV de entrada con una columna 'frase' (y opcionalmente columnas *_manual)."
    )
    parser.add_argument(
        "--output", "-o", default="gpt_classification_results.csv",
        help="CSV de salida con los resultados."
    )
    parser.add_argument(
        "--prompt", "-p", default=DEFAULT_PROMPT_FILE,
        help="Archivo de prompt a usar."
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Evalúa el desempeño comparando con las columnas *_manual."
    )
    return parser.parse_args()


def main():
    """
    Función principal
    """
    args = parse_args()
    print("=== CLASIFICACIÓN DE FRASES CON GPT-4o-mini ===\n")

    # Cargar prompt
    print("Cargando prompt...")
    prompt = load_prompt(args.prompt)

    if not args.input:
        print("Debes indicar un CSV de entrada con --input.")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: el archivo de entrada '{args.input}' no existe.")
        return

    print("Cargando datos...")
    df = pd.read_csv(input_path)

    classify_dataset(df, prompt, validate=args.validate, output_csv=args.output)


if __name__ == "__main__":
    main()
