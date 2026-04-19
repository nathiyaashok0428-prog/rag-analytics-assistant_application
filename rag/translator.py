# =========================================
# MULTILINGUAL TRANSLATOR USING OLLAMA
# English ↔ Portuguese
# =========================================

import re
import requests

from llm_runtime import get_ollama_model, get_ollama_url

try:
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    GoogleTranslator = None

OLLAMA_URL = get_ollama_url()
OLLAMA_MODEL = get_ollama_model()


def clean_translation_output(text):

    cleaned = text.strip()
    cleaned = cleaned.replace("```", "").strip()
    cleaned = re.sub(r"^(translated (portuguese|english) text:)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(portuguese|english)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(this is .*?\)\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(note: .*?\)\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \"'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# =========================================
# ENGLISH → PORTUGUESE
# =========================================

def _translate_with_google(text: str, source: str, target: str) -> str:
    if not text or GoogleTranslator is None:
        return text

    try:
        translated = GoogleTranslator(source=source, target=target).translate(text)
        return clean_translation_output(translated or text)
    except Exception:
        return text


def translate_to_portuguese(text):

    prompt = f"""
Translate the following English text to Portuguese.

Return ONLY translated Portuguese text.
Do not add notes, explanations, options, or commentary.

Text:
{text}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30,
        )

        result = response.json()

        translated_text = clean_translation_output(result["response"])

        return translated_text

    except Exception:
        return _translate_with_google(text, source="en", target="pt")


# =========================================
# PORTUGUESE → ENGLISH
# =========================================

def translate_to_english(text):

    prompt = f"""
Translate the following Portuguese text to English.

Return ONLY translated English text.
Do not add notes, explanations, options, grammar comments, or commentary.
Preserve the meaning of customer review text as closely as possible.

Text:
{text}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30,
        )

        result = response.json()

        translated_text = clean_translation_output(result["response"])

        return translated_text

    except Exception:
        # No key required; works in Streamlit Cloud when Ollama is unavailable.
        return _translate_with_google(text, source="pt", target="en")


# =========================================
# TEST BLOCK
# =========================================

if __name__ == "__main__":

    test_query = "late delivery and damaged product"

    pt_text = translate_to_portuguese(test_query)

    print("\nEnglish:")
    print(test_query)

    print("\nPortuguese:")
    print(pt_text)

    en_text = translate_to_english(pt_text)

    print("\nBack to English:")
    print(en_text)
