# =========================================
# MULTILINGUAL TRANSLATOR USING OLLAMA
# English ↔ Portuguese
# =========================================

import requests
import re

# Ollama local endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"


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
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()

        translated_text = clean_translation_output(result["response"])

        return translated_text

    except Exception as e:

        print("Translation Error:", e)

        # fallback (return original text)
        return text


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
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()

        translated_text = clean_translation_output(result["response"])

        return translated_text

    except Exception as e:

        print("Translation Error:", e)

        return text


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
