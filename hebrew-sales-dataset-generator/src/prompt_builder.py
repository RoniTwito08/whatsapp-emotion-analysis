"""Build prompts for the LLM conversation generator."""

from __future__ import annotations

import random
from typing import Any

from .models import ConversationPlan

FORBIDDEN_PHRASES = [
    # Stock business openers
    "שלום, בכיף. מה בדיוק אתה מחפש?",
    "היי, בהחלט. יש לי כמה אפשרויות להציע",
    "אהלן, כן בהחלט, על מה בדיוק מדובר?",
    "היי, כן בטח. אשמח לעזור",
    "יש לי כמה אפשרויות להציע",
    "הבנתי. המחיר מתחיל בערך מ",
    "היי, בהחלט",
    # Generic mid-conversation questions (appear 40-54x in corpus)
    "יש אפשרות לשלם בכמה תשלומים?",
    "יש אופציה לפריסת תשלומים בלי ריבית.",
    "כמה זמן זה לוקח בערך מתחילת התהליך?",
    "זה תלוי בעומס, אבל בממוצע זה לא לוקח הרבה.",
    "יש לכם ביטוח על העבודה?",
    "כן, יש לנו כיסוי ביטוחי על כל עבודה.",
    "יש לכם צוות קבוע או שזה משתנה כל פעם?",
    "יש לנו צוות קבוע שמלווה את התהליך.",
    "אפשר לשנות תאריך אם משהו יזוז?",
    "בטח, אפשר לשנות תאריך בלי עלות אם מודיעים מראש.",
    "יש לכם ניסיון עם מקרים דומים לשלי?",
    "היה לנו לא מעט מקרים דומים, אין בעיה.",
    "מה קורה אם צריך לבטל בהמשך?",
    "יש הבדל אם מזמינים יותר מיחידה אחת?",
    "יש הנחה קטנה בהזמנה של יותר מיחידה אחת.",
    "יש לכם זמינות השבוע או שצריך לחכות?",
    "כן, יש לנו פתיחות השבוע, אפשר לתאם.",
    "יש עלות נוספת על הובלה או הגעה?",
    "אין עלות נוספת מעבר למחיר שסיכמנו.",
    "אפשר לקבל חשבונית מסודרת?",
    "בטח, חשבונית מסודרת יוצאת אוטומטית.",
    "מי בדיוק מגיע לבצע את זה?",
    "מגיע צוות מקצועי שמכיר את זה טוב.",
    "אפשר לשלוח דוגמאות קודמות?",
    "בטח, אשלח לך כמה דוגמאות מעבודות קודמות.",
    "בטח, אשלח לך כמה דוגמאות מעבודות אחרונות",
    "מה לגבי אחריות על העבודה?",
    "יש אחריות מלאה לתקופה מוגדרת.",
    "אפשר לקבל את זה גם בסוף השבוע?",
    "בטח, יש לנו זמינות גם בסופ״ש.",
    "כמה זמן האחריות בתוקף בערך?",
    "האחריות בתוקף לרוב לכמה חודשים טובים.",
    "צריך להיות בבית כשמגיעים או שאפשר לתאם אחרת?",
    "לא חובה להיות בבית, אפשר לתאם גישה מראש.",
    "יש לכם חבילה מצומצמת יותר במחיר נמוך?",
    "יש לנו גם חבילה מצומצמת יותר במחיר נוח.",
    "יש לנו גם חבילה מצומצמת יותר במחיר נגיש",
    "מה ההבדל בין החבילות שיש לכם?",
    "אפשר לדעת קצת יותר על התהליך?",
    "כמה זמן לוקח מהתחלה ועד סוף?",
    "יש לכם ביקורות או לקוחות שאפשר לראות?",
    "יש לכם אחריות גם על חלקי חילוף",
    "כן, האחריות חלה גם על חלקי חילוף מקוריים",
    "יגיע איש צוות מנוסה שמכיר את זה היטב",
    "אנחנו משתדלים לחזור תוך יום עבודה לכל היותר",
    "האחריות בתוקף בדרך כלל לכמה חודשים טובים",
    "אפשר לתאם גם לסוף שבוע",
    "בטח, יש לנו זמינות גם בסופי שבוע",
    "מי בדיוק יגיע לבצע את זה בפועל",
    "כן, יש לנו כיסוי ביטוחי מסודר על כל עבודה",
    "אפשר לשלם באשראי, בהעברה או במזומן, איך שנוח",
    "בטח, אפשר לקבל טיוטה לעיון לפני חתימה סופית",
    "יש הנחה יפה בהזמנה גדולה יותר",
    "יש לכם כיסוי ביטוחי על העבודה",
    "בהחלט, יש הנחה יפה כשמשלבים כמה שירותים יחד",
    "כן, יש לנו אתר עם תיק עבודות מסודר",
    "יש לכם המלצות מלקוחות מהאזור",
    # Stock closings (heavily repeated by outcome)
    "נראה לי שזה עדיין יקר לי, תודה בכל מקרה.",
    "מובן, שיהיה בהצלחה.",
    "כן, נקבע.",
    "מעולה תודה.",
    "סבבה, שולח עכשיו.",
    "מעולה, בוא נסגור.",
    "צריך ממך רק שם מלא וטלפון לאישור.",
    "מעולה, אני אעדכן מחר.",
    "היי, רק בודק שהכל הגיע",
    "בטח, שולח עכשיו. תעדכן אותי מה חשבת",
    "מעולה, נדבר בקרוב",
    "נדבר בקרוב",
    "אעדכן",
    "סגרנו",
    "תודה רבה",
    "שלום, רציתי לשאול",
    "בשמחה",
    "מעולה, תודה",
    # Generic mid-conversation customer lines
    "רוצה להבין מה כלול במחיר",
    "אני צריך משהו יחסית דחוף",
    "אני לא רוצה להתחייב לפני שאני מבין הכל",
    "חשוב לי שזה יהיה איכותי ושלא אצטרך להתעסק אחר כך",
    "בלי התחייבות? זה דווקא משנה. אפשר לשמוע?",
    "ראיתי גם כמה תגובות פחות טובות, זה קצת הוריד לי.",
    "יש אפשרות בסיסית יותר או פריסה לתשלומים.",
    "ברור, אפשר לקבוע שיחת היכרות קצרה בלי התחייבות.",
]

# Maps each domain to a service category, then each category to prompt guidance.
_DOMAIN_CATEGORY: dict[str, str] = {
    "cleaning": "physical_home", "pest_control": "physical_home",
    "air_conditioning": "physical_home", "electrical": "physical_home",
    "plumbing": "physical_home", "construction": "physical_home",
    "solar": "physical_home", "home_renovation": "physical_home",
    "moving": "physical_home", "storage": "physical_home",
    "interior_design": "physical_home",
    "furniture": "physical_product", "electronics": "physical_product",
    "kitchens": "physical_product", "pet_store": "physical_product",
    "beauty": "personal_service", "hair_salon": "personal_service",
    "dental": "personal_service", "veterinary": "personal_service",
    "fitness": "personal_service", "private_medical": "personal_service",
    "child_development": "personal_service",
    "legal": "professional_service", "insurance": "professional_service",
    "business_consulting": "professional_service", "recruitment": "professional_service",
    "software": "professional_service", "graphic_design": "professional_service",
    "printing": "professional_service",
    "travel": "experience_booking", "hotels": "experience_booking",
    "restaurants": "experience_booking", "events": "experience_booking",
    "photography": "experience_booking",
    "courses": "education", "tutoring": "education",
    "real_estate": "real_estate",
    "car_service": "automotive",
}

_DOMAIN_CATEGORY_GUIDANCE: dict[str, tuple[str, str]] = {
    "physical_home": (
        "שירות פיזי שמגיע לבית. שאלות מתאימות: גישה לדירה, זמינות, מי מגיע, כמה עובדים, רעש/לכלוך, ביטוח, ערבות על העבודה.",
        "אסור לשאול: שאלות על קורסים, תוכן דיגיטלי, מיקום הסניף, ישיבה בקליניקה.",
    ),
    "physical_product": (
        "רכישת מוצר פיזי. שאלות מתאימות: מידות, חומרים, צבעים, מלאי, זמן אספקה, הרכבה, מדיניות החזרה, אחריות יצרן.",
        "אסור לשאול: ביטוח עבודה, מי מגיע לבצע, צוות, עלות הגעה.",
    ),
    "personal_service": (
        "טיפול אישי במקום העסק (סלון, קליניקה, קבינט). שאלות מתאימות: מה כולל הטיפול, כמה זמן, מי המטפל, ניסיון, תוצאות, הכנה מראש, תור חוזר.",
        "אסור לשאול: הגעה הביתה, ביטוח עבודה פיזית, הרכבת ציוד, אחריות על ביצוע.",
    ),
    "professional_service": (
        "שירות מקצועי/ייעוצי. שאלות מתאימות: ניסיון, תהליך עבודה, כמה פגישות/שיחות, מה מקבלים בסוף, תיעוד, סודיות.",
        "אסור לשאול: הגעת טכנאי, ביטוח עבודה פיזית, מי מגיע לבצע, הרכבה, עלות נסיעה.",
    ),
    "experience_booking": (
        "הזמנת חוויה, אירוע או שהייה. שאלות מתאימות: תאריכים, מה כולל, ביטול/שינוי הזמנה, גודל קבוצה, מה להביא, מיקום.",
        "אסור לשאול: ביטוח עבודה, צוות הרכבה, גישה לדירה, מי מגיע לבצע.",
    ),
    "education": (
        "קורס, שיעורים, הדרכה. שאלות מתאימות: תוכן הלימוד, פורמט (זום/פרונטלי), רמת כניסה, תרגולים, תעודה, גמישות בזמנים.",
        "אסור לשאול: ביטוח עבודה, הגעת צוות, הרכבה פיזית, עלות נסיעה.",
    ),
    "real_estate": (
        "מכירה, השכרה או תיווך נדל\"ן. שאלות מתאימות: מיקום, קומה, חניה, גודל, מצב הדירה, עמלת תיווך, זמינות לצפייה.",
        "אסור לשאול: ביטוח עבודה פיזית, הרכבת ציוד.",
    ),
    "automotive": (
        "שירות לרכב. שאלות מתאימות: סוג התקלה, כמה זמן תיקון, חלקי חילוף מקוריים, אחריות, רכב חלופי, בדיקה לפני קנייה.",
        "אסור לשאול: עיצוב פנים, הגעה הביתה לתיקון (אלא אם כן מדובר בשירות גרר).",
    ),
}


def _get_domain_guidance(domain: str) -> str:
    cat = _DOMAIN_CATEGORY.get(domain, "")
    if not cat:
        return ""
    relevant, avoid = _DOMAIN_CATEGORY_GUIDANCE.get(cat, ("", ""))
    if not relevant:
        return ""
    return f"{relevant}\n{avoid}"


TRAJECTORY_INSTRUCTIONS = {
    "high_to_conversion": (
        "הלקוח מגיע עם עניין גבוה, שואל שאלות ממוקדות, ומתקדם בצורה טבעית לסגירת עסקה. "
        "הסיום: הלקוח אומר בפירוש שהוא רוצה להתקדם ומסכים לתנאים."
    ),
    "high_to_price_rejection": (
        "הלקוח מגיע עם עניין גבוה אך נדהם מהמחיר. הוא לא מגלה עניין בהנחות וסוגר את השיחה. "
        "הסיום: הלקוח מצהיר שזה יקר לו מדי ומסיים."
    ),
    "high_to_ghosting": (
        "הלקוח מגיע עם עניין גבוה, שואל שאלות, ואז פשוט מפסיק לענות. "
        "העסק שולח הודעת מעקב אחרונה שנשארת ללא מענה. "
        "הסיום: העסק שולח הודעת מעקב, ואין תגובה מהלקוח."
    ),
    "high_to_delivery_loss": (
        "הלקוח מגיע עם עניין גבוה, אך מגלה שהמועד אינו מתאים לו. "
        "הסיום: הלקוח מסיים כי לא יכול לחכות."
    ),
    "high_to_trust_loss": (
        "הלקוח מגיע בעניין גבוה, אך מעלה שאלות על אמינות, ניסיון, או ביקורות. "
        "הוא מאבד אמון בהדרגה ומסיים בסירוב. "
        "הסיום: הלקוח מצהיר שהוא לא מרגיש בטוח ומסיים."
    ),
    "medium_to_discount_conversion": (
        "הלקוח מגיע בעניין בינוני, מגיב בעדינות לשאלות המחיר, אך נמשך אחרי הצעה מוקטנת. "
        "הסיום: הלקוח מסכים לחבילה קטנה יותר ומתקדם."
    ),
    "medium_to_competitor_loss": (
        "הלקוח מגיע בעניין בינוני ומציין שיש לו הצעה ממתחרה. "
        "למרות נסיון העסק, הוא בוחר במתחרה. "
        "הסיום: הלקוח בוחר במתחרה ומסיים."
    ),
    "medium_to_pending": (
        "הלקוח מגיע בעניין בינוני, שואל שאלות, אך לא מגיע להחלטה. "
        "הוא צריך לחשוב, לשאול, או לחכות. "
        "הסיום: הלקוח מבטיח לחזור אך השיחה מסתיימת ללא החלטה."
    ),
    "low_to_appointment": (
        "הלקוח מגיע בחשדנות או עניין נמוך, אך לאחר הסבר נכון הוא מסכים לתאם פגישה/תור. "
        "הסיום: תאריך ושעה נקבעים בצורה ברורה."
    ),
    "low_to_reengagement": (
        "הלקוח מגיע בעניין נמוך ומאבד עניין אחרי שמיעת המחיר. "
        "העסק מציע אפשרות חלופית והלקוח חוזר לעניין ומבקש לשמוע יותר. "
        "הסיום: הלקוח מבקש פרטים נוספים ומתכנן לחזור."
    ),
}

GENDER_INSTRUCTIONS = {
    "male": (
        "הלקוח הוא גבר. השתמש בצורות דקדוקיות גבריות בלבד לאורך כל השיחה. "
        "לא להשתמש בצורות כפולות כמו מוכן/ה, רוצה/ה, חושב/ת."
    ),
    "female": (
        "הלקוחה היא אישה. השתמש בצורות דקדוקיות נשיות בלבד לאורך כל השיחה. "
        "לא להשתמש בצורות כפולות כמו מוכן/ה, רוצה/ה, חושב/ת."
    ),
}


def build_system_prompt() -> str:
    return (
        "אתה מחולל שיחות WhatsApp עסקיות בעברית לצורכי מחקר. "
        "תפקידך לכתוב טקסט של הודעות בלבד — לא מבנה JSON, לא שדות נוספים. "
        "כל שיחה חייבת להיות טבעית, לא תסריטאית, ולא חזרה על תבניות קיימות. "
        "הישמע כמו שיחה אמיתית בין אנשים אמיתיים."
    )


def build_user_prompt(
    plan: ConversationPlan,
    forbidden_samples: list[str],
    recent_accepted_samples: list[str],
    retry_feedback: str | None = None,
) -> str:
    trajectory_instruction = TRAJECTORY_INSTRUCTIONS.get(plan.trajectory, "")
    gender_instruction = GENDER_INSTRUCTIONS.get(plan.customer_gender, "")

    forbidden_block = "\n".join(f"- {p}" for p in forbidden_samples[:25])
    recent_block = "\n".join(f"- {s}" for s in recent_accepted_samples[:10])

    retry_block = ""
    if retry_feedback:
        retry_block = f"\n\n⚠️ הערה לניסיון זה: {retry_feedback}"

    customer_role_name = "לקוח" if plan.customer_gender == "male" else "לקוחה"
    total_messages = plan.message_count

    domain_guidance = _get_domain_guidance(plan.domain)
    domain_guidance_block = ""
    if domain_guidance:
        domain_guidance_block = f"\n== הנחיות תחום: {plan.domain_he} ==\n{domain_guidance}\n"

    prompt = f"""צור שיחת WhatsApp עסקית בעברית לפי התוכנית הבאה:

== תוכנית ==
תחום: {plan.domain_he}
מוצר/שירות: {plan.product_or_service}
תרחיש: {plan.specific_scenario}

{customer_role_name}:
- פרסונה: {plan.customer_persona}
- מטרה: {plan.customer_goal}
- דאגה עיקרית: {plan.main_concern}
- דאגה משנית: {plan.secondary_concern}
- רמה טכנית: {plan.customer_technical_level}
- דחיפות: {plan.urgency}
- רגישות תקציבית: {plan.budget_sensitivity}
- רמת אמון: {plan.trust_level}

עסק: סגנון {plan.business_style}
הקשר פתיחה: {plan.starting_context}

מסלול: {plan.trajectory}
נקודת מפנה: {plan.turning_point}
סיום: {plan.ending}

מספר הודעות: {total_messages} בדיוק
סגנון כתיבה: {plan.writing_style}
דפוס זמן: {plan.timing_pattern}

== הנחיות ==
{trajectory_instruction}

{gender_instruction}

1. כתוב בדיוק {total_messages} הודעות.
2. התחל עם {customer_role_name} ולחלופין עסק/{customer_role_name} לאורך השיחה.
3. הסיום חייב להתאים למסלול שצוין.
4. אל תכלול שדות כמו timestamp, delay, score — רק role ו-text.
5. כתוב עברית ישראלית טבעית — לא מתורגמת, לא רשמית מדי.
6. אל תפתח עם "שלום, רציתי לשאול" ואל תסיים עם "מעולה, תודה / נדבר בקרוב / אעדכן".
7. כתוב תמיד בצורה דקדוקית אחת — גברית או נשית — ואסור להשתמש בצורות מאוחדות כמו "צריך/ה", "שולח/ת", "בוא/י".
{domain_guidance_block}
== הנחיית סיום ==
הסיום חייב להיות ייחודי וספציפי ל"{plan.product_or_service}" ולמצב הלקוח בשיחה זו.
אסור להשתמש בסיומות גנריות. הסיום צריך לשקף את הפרטים הקונקרטיים של השיחה.

== ביטויים אסורים (אל תשתמש בהם) ==
{forbidden_block}

== דוגמאות מהודעות קיימות (להימנע מהם) ==
{recent_block}
{retry_block}

== פורמט תשובה ==
החזר JSON בלבד, ללא markdown, ללא קוד מעטפה:
{{
  "messages": [
    {{"role": "customer", "text": "..."}},
    {{"role": "business", "text": "..."}}
  ],
  "actual_outcome": "תאר בקצרה את התוצאה בפועל",
  "trajectory_notes": "הסבר קצר על מהלך השיחה"
}}"""

    return prompt


def sample_recent_messages(
    recent_conversations: list[dict],
    n: int = 10,
    rng: random.Random | None = None,
) -> list[str]:
    if rng is None:
        rng = random.Random()
    all_texts = []
    for c in recent_conversations[-30:]:
        for m in c.get("messages", []):
            text = m.get("text", "").strip()
            if text and len(text) > 10:
                all_texts.append(text)
    if len(all_texts) > n:
        return rng.sample(all_texts, n)
    return all_texts
