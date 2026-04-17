import re
import unicodedata
from typing import Final

import requests

# =========================================
# MULTILINGUAL TRANSLATOR
# English ↔ Portuguese
# =========================================

OLLAMA_URL: Final[str] = "http://localhost:11434/api/generate"
MYMEMORY_URL: Final[str] = "https://api.mymemory.translated.net/get"
GOOGLE_TRANSLATE_URL: Final[str] = "https://translate.googleapis.com/translate_a/single"
REQUEST_TIMEOUT_SECONDS: Final[int] = 12
PT_TO_EN_FALLBACK: Final[dict[str, str]] = {
    "entrega": "delivery",
    "atrasada": "late",
    "atraso": "delay",
    "demorou": "took too long",
    "produto": "product",
    "produtos": "products",
    "danificado": "damaged",
    "problema": "problem",
    "problemas": "problems",
    "cliente": "customer",
    "clientes": "customers",
    "pedido": "order",
    "pedi": "ordered",
    "email": "email",
    "responde": "respond",
    "responder": "respond",
    "empresa": "company",
    "fornecedor": "supplier",
    "devolucao": "return",
    "reclamacao": "complaint",
    "nao": "not",
    "nunca": "never",
    "ruim": "bad",
    "e": "and",
    "com": "with",
    "sem": "without",
    "a": "the",
    "o": "the",
    "os": "the",
    "as": "the",
    "um": "a",
    "uma": "a",
    "do": "of the",
    "da": "of the",
    "dos": "of the",
    "das": "of the",
    "de": "of",
    "para": "for",
    "por": "by",
    "sobre": "about",
    "ate": "until",
    "ao": "to the",
    "emails": "emails",
}
EN_TO_PT_FALLBACK: Final[dict[str, str]] = {
    "delivery": "entrega",
    "late": "atrasada",
    "delay": "atraso",
    "product": "produto",
    "products": "produtos",
    "damaged": "danificado",
    "problem": "problema",
    "problems": "problemas",
    "customer": "cliente",
    "customers": "clientes",
    "order": "pedido",
    "email": "email",
    "respond": "responder",
    "company": "empresa",
    "supplier": "fornecedor",
    "return": "devolucao",
    "complaint": "reclamacao",
    "complaints": "reclamacoes",
    "not": "nao",
    "never": "nunca",
    "bad": "ruim",
    "and": "e",
    "with": "com",
    "without": "sem",
    "the": "o",
    "a": "um",
    "of": "de",
    "for": "para",
    "by": "por",
    "about": "sobre",
    "until": "ate",
    "to": "para",
}


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


def _normalize_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _dictionary_translate(text: str, dictionary: dict[str, str]) -> str:
    token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
    translated_tokens: list[str] = []
    for token in token_pattern.findall(text):
        normalized = _normalize_token(token)
        translated_tokens.append(dictionary.get(normalized, token))

    translated = " ".join(translated_tokens)
    translated = re.sub(r"\s+([.,;:!?])", r"\1", translated)
    return translated


def _looks_portuguese(text: str) -> bool:
    normalized = _normalize_token(text)
    hints = [
        " nao ",
        " entrega ",
        " cliente ",
        " produto ",
        " pedido ",
        " atras",
        " reclam",
        " que ",
        " para ",
    ]
    padded = f" {normalized} "
    return any(hint in padded for hint in hints)


def _looks_english(text: str) -> bool:
    normalized = _normalize_token(text)
    hints = [
        " delivery ",
        " customer ",
        " product ",
        " order ",
        " complaint ",
        " and ",
        " with ",
        " not ",
    ]
    padded = f" {normalized} "
    return any(hint in padded for hint in hints)


def _is_reasonable_translation(candidate: str, source_code: str, target_code: str) -> bool:
    if not candidate or not candidate.strip():
        return False
    if target_code == "en" and _looks_portuguese(candidate) and not _looks_english(candidate):
        return False
    if target_code == "pt" and _looks_english(candidate) and not _looks_portuguese(candidate):
        return False
    return True


def _translate_with_ollama(text: str, source_language: str, target_language: str) -> str | None:
    prompt = f"""
Translate the following {source_language} text to {target_language}.

Return ONLY translated {target_language} text.
Do not add notes, explanations, options, or commentary.

Text:
{text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    translated_text = clean_translation_output(payload.get("response", ""))
    return translated_text or None


def _translate_with_mymemory(text: str, source_code: str, target_code: str) -> str | None:
    response = requests.get(
        MYMEMORY_URL,
        params={"q": text, "langpair": f"{source_code}|{target_code}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    translated_text = clean_translation_output(
        payload.get("responseData", {}).get("translatedText", "")
    )
    return translated_text or None


def _translate_with_google(text: str, source_code: str, target_code: str) -> str | None:
    response = requests.get(
        GOOGLE_TRANSLATE_URL,
        params={
            "client": "gtx",
            "sl": source_code,
            "tl": target_code,
            "dt": "t",
            "q": text,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    segments = payload[0] if payload else []
    translated_text = "".join(
        segment[0] for segment in segments if isinstance(segment, list) and segment
    )
    translated_text = clean_translation_output(translated_text)
    return translated_text or None


def _translate(
    text: str,
    source_language: str,
    target_language: str,
    source_code: str,
    target_code: str,
    fallback_dictionary: dict[str, str],
) -> str:
    if not text or not text.strip():
        return text

    try:
        translated = _translate_with_ollama(text, source_language, target_language)
        if translated and _is_reasonable_translation(translated, source_code, target_code):
            return translated
    except Exception:
        pass

    try:
        translated = _translate_with_mymemory(text, source_code, target_code)
        if translated and _is_reasonable_translation(translated, source_code, target_code):
            return translated
    except Exception:
        pass

    try:
        translated = _translate_with_google(text, source_code, target_code)
        if translated and _is_reasonable_translation(translated, source_code, target_code):
            return translated
    except Exception:
        pass

    return _dictionary_translate(text, fallback_dictionary)


# =========================================
# ENGLISH → PORTUGUESE
# =========================================
def translate_to_portuguese(text):
    return _translate(
        text=text,
        source_language="English",
        target_language="Portuguese",
        source_code="en",
        target_code="pt",
        fallback_dictionary=EN_TO_PT_FALLBACK,
    )


# =========================================
# PORTUGUESE → ENGLISH
# =========================================

def translate_to_english(text):
    return _translate(
        text=text,
        source_language="Portuguese",
        target_language="English",
        source_code="pt",
        target_code="en",
        fallback_dictionary=PT_TO_EN_FALLBACK,
    )


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
