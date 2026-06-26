import re
import unicodedata


EMERGENCY_PATTERNS = [
    r"\bsuicide\b",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bself harm\b",
    r"\bmarna man\b",
    r"\bmarna chahanchu\b",
    r"\baatmahatya\b",
    r"आत्महत्या",
    r"मर्न मन",
    r"आफैलाई मार",
    r"बाँच्न मन छैन",
    r"मर्न चाहन्छु",
]


ROMAN_NEPALI_HINTS = {
    "mero": "मेरो",
    "malai": "मलाई",
    "padhai": "पढाइ",
    "padhna": "पढ्न",
    "tanab": "तनाव",
    "chinta": "चिन्ता",
    "nindra": "निन्द्रा",
    "exam": "परीक्षा",
    "college": "कलेज",
    "teacher": "शिक्षक",
    "sathi": "साथी",
    "dukha": "दुख",
    "khusi": "खुसी",
    "man": "मन",
}


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def looks_romanized_nepali(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered.split() for token in ROMAN_NEPALI_HINTS)


def simple_roman_hint_rewrite(text: str) -> str:
    words = text.split()
    converted = [ROMAN_NEPALI_HINTS.get(word.lower().strip(".,?!"), word) for word in words]
    return " ".join(converted)


def is_emergency(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in EMERGENCY_PATTERNS)


def emergency_response() -> str:
    return (
        "तपाईं अहिले एक्लै बस्नु पर्दैन। कृपया तुरुन्तै नजिकको विश्वासिलो व्यक्ति, परिवार, "
        "कलेज काउन्सेलर वा स्थानीय आपतकालीन सेवामा सम्पर्क गर्नुहोस्। यदि तत्काल जोखिम छ भने "
        "अहिले नै नजिकको अस्पताल वा आपतकालीन नम्बरमा फोन गर्नुहोस्।"
    )

