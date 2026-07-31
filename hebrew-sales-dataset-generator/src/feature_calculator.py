"""Calculate deterministic behavioral features from Hebrew message text."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "♀-♂"
    "☀-⭕"
    "‍"
    "⏏"
    "⏩"
    "⌚"
    "️"
    "〰"
    "]+",
    flags=re.UNICODE,
)

_PRICE_KEYWORDS = re.compile(
    r"(?:"
    r"ש[\"״]ח|₪|שקל|שקלים|אלף\s*ש|"
    r"מחיר|עלות|עלה|עולה|תמחור|תשלום|תשלומים|"
    r"מקדמה|מקדמות|תשלום\s*ראשון|"
    r"הנחה|הנחות|הוזלה|"
    r"חבילה|חבילות|"
    r"תקציב|"
    r"כמה\s*(?:זה|עולה|עול|עלה|תעלה|יעלה|עולים)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_PRICE_AMOUNT_RE = re.compile(
    r"(?:"
    r"\d[\d,\.]*\s*(?:ש[\"״]ח|₪|שקל|שקלים|אלף|k|K)|"
    r"(?:₪|ש[\"״]ח)\s*\d[\d,\.]*"
    r")",
    re.UNICODE,
)

_NOT_PRICE_RE = re.compile(
    r"(?:"
    r"\b(?:0[89]\d{1,}-?\d{3,}|\d{2,4}-\d{3,4}-\d{3,4})\b|"
    r"(?:ינואר|פברואר|מרץ|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר)"
    r")",
    re.UNICODE,
)

_NEGOTIATION_KEYWORDS = re.compile(
    r"(?:"
    r"הנחה|להוריד\s*(?:את\s*)?המחיר|הוזלה|"
    r"הצעת\s*מחיר\s*(?:אחרת|יותר\s*טובה)|"
    r"מתחרה|הצעה\s*(?:יותר\s*)?(?:זולה|טובה)\s*(?:מ|ממקום|ממתחרה)|"
    r"לפצל\s*(?:את\s*)?התשלום|תשלומים\s*(?:נוספים|גמישים)|"
    r"להוריד\s*(?:מ|מה)(?:חבילה|שירות)|"
    r"להסיר\s*(?:מ|מה)(?:חבילה|שירות)|"
    r"בלי\s+(?:ה)?(?:אחריות|התקנה|הובלה|שירות\s*נוסף)|"
    r"קיבלתי\s*(?:הצעה|מחיר)\s*(?:יותר\s*)?(?:טובה|זולה|נמוכה)|"
    r"מציעים\s*לי\s*(?:פחות|יותר\s*זול)|"
    r"ניתן\s*(?:להוריד|לשנות|לצמצם)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_OBJECTION_KEYWORDS = re.compile(
    r"(?:"
    r"יקר\s*(?:לי|מדי|מאוד)?|"
    r"לא\s+(?:בטוח|בטוחה|בטוחים|מוכן|מוכנה|מוכנים)\s*(?:ש|ל|ב)|"
    r"לא\s+(?:מתאים|מתאימה|מתאים\s*לי)|"
    r"לא\s+(?:בהכרח|כל\s*כך|ממש)\s*צריך|"
    r"מהסס|מהססת|מהססים|"
    r"לא\s+(?:יודע|יודעת|בטוח|בטוחה)\s*אם|"
    r"קצת\s+(?:מפחד|מפחדת|חושש|חוששת|מודאג|מודאגת)|"
    r"לא\s+(?:ממהר|ממהרת|דחוף\s*לי)|"
    r"נשמע\s+(?:הרבה|יקר|מסובך|מורכב|גדול)|"
    r"צריך\s+(?:לחשוב|לבדוק|לשאול)|"
    r"(?:פחות|לא)\s+(?:מתאים|מתאימה)\s+לי|"
    r"אין\s+(?:לי|לנו)\s+(?:את\s+)?(?:התקציב|הכסף|הזמן)|"
    r"(?:פחות|לא\s*)(?:בדחיפות|דחוף\s+לי)|"
    r"מתלבט|מתלבטת|"
    r"עדיין\s+לא\s+(?:החלטתי|החלטנו)|"
    r"יש\s+לי\s+(?:ספקות|חששות|שאלות)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_COMMITMENT_KEYWORDS = re.compile(
    r"(?:"
    r"(?:נ)?(?:סגור|נסגור|נסגרנו|סגרנו)|"
    r"(?:מ)?(?:אשר|מאשר|מאשרת|מאושר)\s*(?:את\s*ה)?(?:הצעה|עסקה|קנייה|הזמנה)|"
    r"(?:אני\s+)?(?:שו)?(?:לח|שולח|שולחת)\s*(?:לך\s*)?(?:את\s*)?(?:הפרטים|המסמכים|הסכם|התשלום|מקדמה)|"
    r"(?:מ)?(?:זמין|זמינה|מזמין|מזמינה)\s*(?:אותך|עכשיו)|"
    r"(?:נ)?(?:קבע|נקבע|קבענו)\s*(?:פגישה|תור|ביקור|פגישת\s*הכרות|זמן)|"
    r"(?:אני\s+)?(?:מ)?(?:אשלם|שולם|אשלח\s*תשלום)|"
    r"בוא\s+(?:נ)?(?:סגור|נתקדם|נקבע)|"
    r"אני\s+(?:מ)?(?:סכים|סכימה|לוקח|לוקחת)|"
    r"(?:ה)?הזמנה\s*(?:שלי\s*)?(?:מאושרת|נשלחה|נשלח)|"
    r"(?:ה)?(?:תור|פגישה)\s*(?:מ)?(?:אושר|נקבע|עומד)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_DELAY_KEYWORDS = re.compile(
    r"(?:"
    r"(?:אח)?(?:זור|אחזור|נחזור)\s*אליך?\s*(?:מחר|בקרוב|בשבוע|בימים\s*הקרובים)|"
    r"(?:אני\s+)?(?:צריך|צריכה|נצטרך)\s+(?:לבדוק|לשאול|לחשוב|להתייעץ|לדבר)|"
    r"(?:נ)?(?:דבר|נדבר)\s*(?:אחרי|לאחר|עם\s*סיום|אחרי\s+(?:ה)?חג|אחרי\s+(?:ה)?חגים|בשבוע\s+הבא|בחודש\s+הבא)|"
    r"(?:תן|תני)\s*לי\s*(?:כמה\s*)?(?:יום|ימים|שעות|זמן|לחשוב)|"
    r"(?:אני\s+)?(?:ממ)?(?:תן|מחכה|מחכה\s+לתשובה)\s+(?:מ(?:ה)?|ל(?:ה)?)?(?:מנהל|שותף|אישה|בעל|הורים)|"
    r"(?:ה)?חג(?:ים)?\s+(?:מפריע|קצת\s+מפריע|עובר|עובד)|"
    r"אחרי\s+(?:ה)?(?:חופש|חג|קיץ|חגים)|"
    r"(?:ב)?(?:שבוע|חודש|שנה)\s+(?:הבא|הבאה)|"
    r"(?:בהמשך|בהמשך\s+השבוע|מאוחר\s+יותר)|"
    r"(?:תציי)?(?:ן|נ)?\s*(?:לי|לנו)\s*(?:כמה\s*)?(?:ימים?|זמן)|"
    r"לא\s+(?:ממהר|ממהרת)\s+(?:לרגע|כרגע)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_DELAY_FALSE_POSITIVES = re.compile(
    r"(?:רגע\s+(?:אחד|אחד\s+בבקשה)|רק\s+רגע|חכה\s+רגע|שניה)",
    re.IGNORECASE | re.UNICODE,
)


def count_questions(text: str) -> int:
    return text.count("?")


def count_emojis(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


def detect_contains_price(text: str) -> bool:
    if _NOT_PRICE_RE.search(text) and not _PRICE_AMOUNT_RE.search(text):
        return False
    return bool(_PRICE_KEYWORDS.search(text) or _PRICE_AMOUNT_RE.search(text))


def detect_contains_negotiation(text: str) -> bool:
    return bool(_NEGOTIATION_KEYWORDS.search(text))


def detect_contains_objection(text: str) -> bool:
    return bool(_OBJECTION_KEYWORDS.search(text))


def detect_contains_commitment(text: str) -> bool:
    return bool(_COMMITMENT_KEYWORDS.search(text))


def detect_contains_delay_signal(text: str) -> bool:
    if _DELAY_FALSE_POSITIVES.search(text):
        rest = _DELAY_FALSE_POSITIVES.sub("", text)
        return bool(_DELAY_KEYWORDS.search(rest))
    return bool(_DELAY_KEYWORDS.search(text))


def calculate_behavioral_features(text: str) -> dict[str, Any]:
    return {
        "message_length_chars": len(text),
        "question_count": count_questions(text),
        "emoji_count": count_emojis(text),
        "contains_price": detect_contains_price(text),
        "contains_negotiation": detect_contains_negotiation(text),
        "contains_objection": detect_contains_objection(text),
        "contains_commitment": detect_contains_commitment(text),
        "contains_delay_signal": detect_contains_delay_signal(text),
    }
