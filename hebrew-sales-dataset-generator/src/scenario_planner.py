"""Generate diverse conversation plans before calling the LLM."""

from __future__ import annotations

import random
from typing import Any

from .models import ConversationPlan

DOMAIN_CATALOG: list[dict[str, Any]] = [
    {
        "domain": "furniture",
        "domain_he": "ריהוט",
        "scenarios": [
            {"product": "פינת אוכל", "concerns": ["גודל", "חומר", "אחריות", "משלוח"]},
            {"product": "ספה פינתית", "concerns": ["צבע", "בד", "מידה", "מחיר"]},
            {"product": "ארון בגדים", "concerns": ["גובה", "מגירות", "סגנון"]},
            {"product": "מיטה זוגית", "concerns": ["מידה", "ראש מיטה", "חומר", "מחיר"]},
            {"product": "שולחן עבודה", "concerns": ["גודל", "חומר", "אחסון"]},
        ],
    },
    {
        "domain": "kitchens",
        "domain_he": "מטבחים",
        "scenarios": [
            {"product": "ארונות מטבח", "concerns": ["חומר", "עיצוב", "כיור", "מחיר"]},
            {"product": "אי למטבח", "concerns": ["גודל", "אחסון", "התקנה"]},
            {"product": "חידוש מטבח", "concerns": ["זמן ביצוע", "חומרים", "מחיר"]},
        ],
    },
    {
        "domain": "air_conditioning",
        "domain_he": "מיזוג",
        "scenarios": [
            {"product": "מזגן אינוורטר", "concerns": ["הספק", "התקנה", "אחריות", "מחיר"]},
            {"product": "תיקון מזגן", "concerns": ["אבחון", "חלקים", "עלות"]},
            {"product": "ניקוי מזגן", "concerns": ["תדירות", "עלות", "זמינות"]},
        ],
    },
    {
        "domain": "real_estate",
        "domain_he": "נדל״ן",
        "scenarios": [
            {"product": "דירה להשכרה", "concerns": ["שכירות", "ממ\"ד", "חניה", "מיקום"]},
            {"product": "דירה לקנייה", "concerns": ["מחיר", "קומה", "שטח", "שכנים"]},
            {"product": "משרד להשכרה", "concerns": ["מחיר", "חניה", "גודל", "חוזה"]},
        ],
    },
    {
        "domain": "fitness",
        "domain_he": "כושר",
        "scenarios": [
            {"product": "אימון אישי", "concerns": ["תוצאות", "זמינות", "מחיר", "ניסיון"]},
            {"product": "מנוי לחדר כושר", "concerns": ["ציוד", "שעות", "מחיר", "מיקום"]},
            {"product": "תוכנית תזונה וכושר", "concerns": ["גמישות", "תוצאות", "זמן"]},
        ],
    },
    {
        "domain": "beauty",
        "domain_he": "קוסמטיקה",
        "scenarios": [
            {"product": "טיפול פנים", "concerns": ["תוצאות", "מוצרים", "תדירות", "מחיר"]},
            {"product": "הסרת שיער בלייזר", "concerns": ["כאב", "יעילות", "מחיר", "טיפולים"]},
            {"product": "מניקור ופדיקור", "concerns": ["חומרים", "עמידות", "מחיר"]},
        ],
    },
    {
        "domain": "photography",
        "domain_he": "צילום",
        "scenarios": [
            {"product": "צילום חתונה", "concerns": ["חוויה", "עריכה", "מחיר", "זמינות"]},
            {"product": "צילומי בר מצווה", "concerns": ["סגנון", "מחיר", "וידאו"]},
            {"product": "צילומי תדמית לעסק", "concerns": ["סגנון", "זמן", "תוצאה"]},
        ],
    },
    {
        "domain": "events",
        "domain_he": "אירועים",
        "scenarios": [
            {"product": "אולם אירועים", "concerns": ["קיבולת", "חניה", "קייטרינג", "מחיר"]},
            {"product": "הפקת אירוע קטן", "concerns": ["ריהוט", "תאורה", "עלות"]},
            {"product": "אירוע ילדים", "concerns": ["מקום", "בידור", "אוכל", "מחיר"]},
        ],
    },
    {
        "domain": "insurance",
        "domain_he": "ביטוח",
        "scenarios": [
            {"product": "ביטוח רכב", "concerns": ["כיסוי", "פרמיה", "השתתפות עצמית", "חברה"]},
            {"product": "ביטוח דירה", "concerns": ["כיסוי", "עלות", "תנאים"]},
            {"product": "ביטוח בריאות", "concerns": ["כיסוי", "פרמיה", "בריאות טרום"]},
            {"product": "ביטוח חיים", "concerns": ["סכום", "פרמיה", "תנאים"]},
        ],
    },
    {
        "domain": "car_service",
        "domain_he": "רכב",
        "scenarios": [
            {"product": "טיפול שנתי לרכב", "concerns": ["עלות", "זמן", "חלקים מקוריים"]},
            {"product": "בדיקה לפני קנייה", "concerns": ["יסודיות", "עלות", "זמינות"]},
            {"product": "תיקון מכונאי", "concerns": ["אבחון", "עלות", "אחריות"]},
            {"product": "החלפת שמן", "concerns": ["סוג שמן", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "courses",
        "domain_he": "קורסים",
        "scenarios": [
            {"product": "קורס עיצוב גרפי", "concerns": ["תוכן", "משך", "מחיר", "תעסוקה"]},
            {"product": "קורס שפה", "concerns": ["רמה", "מורה", "לוח זמנים"]},
            {"product": "קורס להסמכה מקצועית", "concerns": ["הכרה", "מחיר", "תוצאה"]},
        ],
    },
    {
        "domain": "moving",
        "domain_he": "הובלות",
        "scenarios": [
            {"product": "הובלת דירה", "concerns": ["עלות", "זמינות", "אריזה", "ביטוח"]},
            {"product": "הובלת רהיטים כבדים", "concerns": ["צוות", "עלות", "ביטוח"]},
            {"product": "אחסנת ריהוט", "concerns": ["גודל", "עלות", "אבטחה"]},
        ],
    },
    {
        "domain": "veterinary",
        "domain_he": "וטרינריה",
        "scenarios": [
            {"product": "בדיקה שנתית לכלב", "concerns": ["ניסיון", "עלות", "חיסונים"]},
            {"product": "עיקור חתול", "concerns": ["ניסיון", "עלות", "טיפול אחרי"]},
            {"product": "ייעוץ תזונה לחיה", "concerns": ["מזון", "עלות", "מעקב"]},
        ],
    },
    {
        "domain": "dental",
        "domain_he": "רפואת שיניים",
        "scenarios": [
            {"product": "ניקוי שיניים מקצועי", "concerns": ["כאב", "עלות", "זמינות"]},
            {"product": "כתרים וגשרים", "concerns": ["עלות", "זמן", "חומר"]},
            {"product": "ייעוץ ליישור שיניים", "concerns": ["סוג", "עלות", "משך"]},
        ],
    },
    {
        "domain": "legal",
        "domain_he": "שירותים משפטיים",
        "scenarios": [
            {"product": "ייעוץ חוזה שכירות", "concerns": ["עלות", "מומחיות", "מהירות"]},
            {"product": "ייצוג בסכסוך עסקי", "concerns": ["ניסיון", "שכ\"ט", "אסטרטגיה"]},
            {"product": "הסכם ממון", "concerns": ["מחיר", "תהליך", "זמן"]},
        ],
    },
    {
        "domain": "cleaning",
        "domain_he": "ניקיון",
        "scenarios": [
            {"product": "ניקיון דירה", "concerns": ["תדירות", "עלות", "מוצרים", "אמינות"]},
            {"product": "ניקיון משרדים", "concerns": ["לו\"ז", "עלות", "צוות"]},
            {"product": "ניקיון לאחר שיפוץ", "concerns": ["עלות", "זמינות", "יסודיות"]},
        ],
    },
    {
        "domain": "pest_control",
        "domain_he": "הדברה",
        "scenarios": [
            {"product": "הדברת ג'וקים", "concerns": ["בטיחות", "עלות", "יעילות"]},
            {"product": "הדברת עכברים", "concerns": ["שיטה", "עלות", "מעקב"]},
            {"product": "הדברה מניעתית", "concerns": ["תדירות", "עלות", "כימיקלים"]},
        ],
    },
    {
        "domain": "electronics",
        "domain_he": "אלקטרוניקה",
        "scenarios": [
            {"product": "תיקון מחשב נייד", "concerns": ["אבחון", "עלות", "זמן", "אחריות"]},
            {"product": "התקנת מצלמות אבטחה", "concerns": ["כמות", "עלות", "ניטור"]},
            {"product": "בניית מחשב לגיימינג", "concerns": ["תקציב", "מפרט", "ביצועים"]},
        ],
    },
    {
        "domain": "travel",
        "domain_he": "תיירות",
        "scenarios": [
            {"product": "חבילת נופש לאירופה", "concerns": ["עלות", "תאריכים", "מלון", "טיסות"]},
            {"product": "טיול מאורגן לאסיה", "concerns": ["לוח זמנים", "עלות", "אוכל"]},
            {"product": "טיסה זולה", "concerns": ["מחיר", "תאריכים", "חברה"]},
        ],
    },
    {
        "domain": "home_renovation",
        "domain_he": "שיפוצים",
        "scenarios": [
            {"product": "שיפוץ מטבח", "concerns": ["עלות", "זמן", "קבלן", "חומרים"]},
            {"product": "בניית ממ\"ד", "concerns": ["עלות", "היתר", "זמן"]},
            {"product": "שיפוץ חדר אמבטיה", "concerns": ["עלות", "חומרים", "זמן ביצוע"]},
        ],
    },
    {
        "domain": "hotels",
        "domain_he": "מלונאות",
        "scenarios": [
            {"product": "חדר זוגי לסוף שבוע", "concerns": ["מחיר", "ארוחת בוקר", "בריכה"]},
            {"product": "אירוע חברה במלון", "concerns": ["קיבולת", "ציוד", "מחיר"]},
        ],
    },
    {
        "domain": "hair_salon",
        "domain_he": "מספרה",
        "scenarios": [
            {"product": "צביעת שיער", "concerns": ["צבע", "עלות", "עמידות"]},
            {"product": "תספורת ועיצוב", "concerns": ["סגנון", "עלות", "זמן"]},
            {"product": "קרטין ויישור", "concerns": ["עמידות", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "tutoring",
        "domain_he": "שיעורים פרטיים",
        "scenarios": [
            {"product": "שיעורי מתמטיקה לבגרות", "concerns": ["שיטה", "תוצאות", "עלות"]},
            {"product": "שיעורי אנגלית", "concerns": ["רמה", "גישה", "עלות", "זמינות"]},
            {"product": "הכנה לפסיכומטרי", "concerns": ["תוכנית", "עלות", "הצלחה"]},
        ],
    },
    {
        "domain": "pet_store",
        "domain_he": "חנות חיות",
        "scenarios": [
            {"product": "אביזרי אקווריום", "concerns": ["מיני דגים", "ציוד", "מחיר"]},
            {"product": "מזון לכלב", "concerns": ["איכות", "מחיר", "כמות"]},
            {"product": "כלוב לציפור", "concerns": ["גודל", "חומר", "מחיר"]},
        ],
    },
    {
        "domain": "software",
        "domain_he": "תוכנה",
        "scenarios": [
            {"product": "אפליקציה לעסק", "concerns": ["פיצ'רים", "עלות", "תחזוקה", "זמן"]},
            {"product": "אתר תדמית", "concerns": ["עיצוב", "SEO", "עלות", "קוד"]},
            {"product": "מערכת CRM", "concerns": ["הטמעה", "הדרכה", "עלות", "תמיכה"]},
        ],
    },
    {
        "domain": "construction",
        "domain_he": "בנייה",
        "scenarios": [
            {"product": "בניית פרגולה", "concerns": ["חומר", "עלות", "היתר", "גודל"]},
            {"product": "שיפוץ מרפסת", "concerns": ["חומר", "עלות", "זמן"]},
            {"product": "ריצוף חיצוני", "concerns": ["חומר", "עלות", "ניקוז"]},
        ],
    },
    {
        "domain": "restaurants",
        "domain_he": "מסעדנות",
        "scenarios": [
            {"product": "אירוע פרטי במסעדה", "concerns": ["תפריט", "עלות", "קיבולת"]},
            {"product": "קייטרינג לאירוע", "concerns": ["תפריט", "עלות", "שירות"]},
            {"product": "ארוחה זוגית מיוחדת", "concerns": ["מנות", "עלות", "שעות"]},
        ],
    },
    {
        "domain": "electrical",
        "domain_he": "חשמלאות",
        "scenarios": [
            {"product": "התקנת נקודות חשמל", "concerns": ["כמות", "עלות", "אישור"]},
            {"product": "תיקון תקלה חשמלית", "concerns": ["אבחון", "עלות", "בטיחות"]},
            {"product": "חיווט לפרויקט בנייה", "concerns": ["מפרט", "עלות", "תאום"]},
        ],
    },
    {
        "domain": "plumbing",
        "domain_he": "אינסטלציה",
        "scenarios": [
            {"product": "תיקון נזילה", "concerns": ["דחיפות", "עלות", "נזק"]},
            {"product": "התקנת דוד שמש", "concerns": ["סוג", "עלות", "אחריות"]},
            {"product": "החלפת ברזים", "concerns": ["מותג", "עלות", "התקנה"]},
        ],
    },
    {
        "domain": "interior_design",
        "domain_he": "עיצוב פנים",
        "scenarios": [
            {"product": "עיצוב דירה חדשה", "concerns": ["סגנון", "עלות", "לוח זמנים"]},
            {"product": "ייעוץ עיצובי", "concerns": ["שעות", "עלות", "גישה"]},
            {"product": "עיצוב חדר ילדים", "concerns": ["בטיחות", "עלות", "סגנון"]},
        ],
    },
    {
        "domain": "cybersecurity",
        "domain_he": "אבטחת מידע",
        "scenarios": [
            {"product": "בדיקת חדירות", "concerns": ["היקף", "עלות", "דו\"ח", "זמן"]},
            {"product": "ייעוץ אבטחה לחברה", "concerns": ["מומחיות", "עלות", "SLA"]},
            {"product": "הכשרת עובדים בסייבר", "concerns": ["תוכן", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "accounting",
        "domain_he": "הנהלת חשבונות",
        "scenarios": [
            {"product": "הנהלת חשבונות לעסק קטן", "concerns": ["שירות", "עלות", "תכנות"]},
            {"product": "ייעוץ מס", "concerns": ["ניסיון", "עלות", "השבת מס"]},
            {"product": "הכנת דוח שנתי", "concerns": ["מהירות", "עלות", "דיוק"]},
        ],
    },
    {
        "domain": "mortgage",
        "domain_he": "משכנתאות",
        "scenarios": [
            {"product": "ייעוץ משכנתא לדירה ראשונה", "concerns": ["ריבית", "עלות", "מסלול"]},
            {"product": "מחזור משכנתא", "concerns": ["חיסכון", "עלות", "תהליך"]},
        ],
    },
    {
        "domain": "physiotherapy",
        "domain_he": "פיזיותרפיה",
        "scenarios": [
            {"product": "טיפול בכאבי גב", "concerns": ["ניסיון", "עלות", "כמות טיפולים"]},
            {"product": "שיקום לאחר ניתוח", "concerns": ["תוכנית", "עלות", "זמינות"]},
        ],
    },
    {
        "domain": "nutrition",
        "domain_he": "תזונה",
        "scenarios": [
            {"product": "ייעוץ תזונה לירידה במשקל", "concerns": ["שיטה", "עלות", "מעקב"]},
            {"product": "תזונה לספורטאים", "concerns": ["מעקב", "תוספים", "עלות"]},
        ],
    },
    {
        "domain": "dog_training",
        "domain_he": "אילוף כלבים",
        "scenarios": [
            {"product": "אילוף גור", "concerns": ["שיטה", "עלות", "תוצאות"]},
            {"product": "טיפול בבעיות התנהגות", "concerns": ["ניסיון", "עלות", "הצלחה"]},
        ],
    },
    {
        "domain": "solar",
        "domain_he": "סולאר",
        "scenarios": [
            {"product": "מערכת סולאר לבית", "concerns": ["ייצור", "עלות", "החזר השקעה"]},
            {"product": "סולאר לעסק", "concerns": ["הספק", "עלות", "חיבור לרשת"]},
        ],
    },
    {
        "domain": "recruitment",
        "domain_he": "גיוס עובדים",
        "scenarios": [
            {"product": "גיוס מנהל מכירות", "concerns": ["מאגר", "עלות", "זמן"]},
            {"product": "גיוס לתפקיד טכני", "concerns": ["מומחיות", "עלות", "מועמדים"]},
        ],
    },
    {
        "domain": "career_coaching",
        "domain_he": "קואצ'ינג קריירה",
        "scenarios": [
            {"product": "ייעוץ קריירה לשינוי תחום", "concerns": ["גישה", "עלות", "תוצאות"]},
            {"product": "הכנה לראיון עבודה", "concerns": ["שיטה", "עלות", "הצלחה"]},
        ],
    },
    {
        "domain": "driving_lessons",
        "domain_he": "שיעורי נהיגה",
        "scenarios": [
            {"product": "שיעורי נהיגה לרישיון", "concerns": ["מחיר לשיעור", "זמינות", "מורה"]},
            {"product": "שיפור נהיגה", "concerns": ["מחיר", "מספר שיעורים", "גישה"]},
        ],
    },
    {
        "domain": "music_lessons",
        "domain_he": "שיעורי מוזיקה",
        "scenarios": [
            {"product": "שיעורי גיטרה", "concerns": ["רמה", "מחיר", "מיקום"]},
            {"product": "שיעורי פסנתר לילדים", "concerns": ["גיל", "מחיר", "זמינות"]},
        ],
    },
    {
        "domain": "translation",
        "domain_he": "תרגום",
        "scenarios": [
            {"product": "תרגום מסמך משפטי", "concerns": ["דיוק", "מחיר", "זמן"]},
            {"product": "תרגום אתר", "concerns": ["שפות", "מחיר", "SEO"]},
        ],
    },
    {
        "domain": "graphic_design",
        "domain_he": "עיצוב גרפי",
        "scenarios": [
            {"product": "עיצוב לוגו לעסק", "concerns": ["סגנון", "עלות", "זכויות"]},
            {"product": "עיצוב פרסומת", "concerns": ["פורמט", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "printing",
        "domain_he": "הדפסה",
        "scenarios": [
            {"product": "הדפסת כרטיסי ביקור", "concerns": ["כמות", "עלות", "עיצוב"]},
            {"product": "הדפסת קטלוג מוצרים", "concerns": ["עלות", "נייר", "זמן"]},
        ],
    },
    {
        "domain": "marketing",
        "domain_he": "שיווק",
        "scenarios": [
            {"product": "ניהול קמפיין פייסבוק", "concerns": ["תקציב", "ROI", "עלות שירות"]},
            {"product": "קידום אורגני SEO", "concerns": ["תוצאות", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "social_media",
        "domain_he": "סושיאל מדיה",
        "scenarios": [
            {"product": "ניהול אינסטגרם לעסק", "concerns": ["תוכן", "עלות", "תוצאות"]},
            {"product": "פרסום ממומן", "concerns": ["תקציב", "קהל יעד", "עלות"]},
        ],
    },
    {
        "domain": "video_production",
        "domain_he": "הפקת וידאו",
        "scenarios": [
            {"product": "סרטון תדמית לחברה", "concerns": ["אורך", "עלות", "עריכה"]},
            {"product": "וידאו לרשתות חברתיות", "concerns": ["סגנון", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "smart_home",
        "domain_he": "בית חכם",
        "scenarios": [
            {"product": "מערכת תאורה חכמה", "concerns": ["תאימות", "עלות", "התקנה"]},
            {"product": "מצלמות אבטחה", "concerns": ["רזולוציה", "אחסון", "עלות"]},
        ],
    },
    {
        "domain": "business_consulting",
        "domain_he": "ייעוץ עסקי",
        "scenarios": [
            {"product": "ייעוץ אסטרטגי לסטארטאפ", "concerns": ["ניסיון", "עלות", "תוצאות"]},
            {"product": "ייעוץ לשיפור תהליכים", "concerns": ["שיטה", "עלות", "זמן"]},
        ],
    },
    {
        "domain": "storage",
        "domain_he": "אחסנה",
        "scenarios": [
            {"product": "מחסן לריהוט", "concerns": ["גודל", "עלות חודשית", "אבטחה"]},
            {"product": "אחסנת מסמכים עסקיים", "concerns": ["ארגון", "עלות", "נגישות"]},
        ],
    },
    {
        "domain": "child_development",
        "domain_he": "התפתחות הילד",
        "scenarios": [
            {"product": "הערכה התפתחותית", "concerns": ["ניסיון", "עלות", "דו\"ח"]},
            {"product": "טיפול בקלינאית תקשורת", "concerns": ["מספר טיפולים", "עלות", "קופ\"ח"]},
        ],
    },
    {
        "domain": "sports_classes",
        "domain_he": "חוגי ספורט",
        "scenarios": [
            {"product": "חוג שחייה לילדים", "concerns": ["גיל", "עלות", "לוח זמנים"]},
            {"product": "שיעורי יוגה", "concerns": ["רמה", "עלות", "מיקום"]},
        ],
    },
    {
        "domain": "appliance_repair",
        "domain_he": "תיקון מכשירי חשמל",
        "scenarios": [
            {"product": "תיקון מכונת כביסה", "concerns": ["אבחון", "עלות", "אחריות"]},
            {"product": "תיקון מקרר", "concerns": ["תקלה", "עלות", "חלקים"]},
        ],
    },
    {
        "domain": "private_medical",
        "domain_he": "רפואה פרטית",
        "scenarios": [
            {"product": "ייעוץ רפואי פרטי", "concerns": ["מומחיות", "עלות", "זמינות"]},
            {"product": "בדיקות דם בביתי", "concerns": ["שירות", "עלות", "מהירות תוצאות"]},
        ],
    },
    {
        "domain": "university_enrollment",
        "domain_he": "הרשמה לאוניברסיטה",
        "scenarios": [
            {"product": "הכוונה לתואר מתאים", "concerns": ["אפשרויות", "דרישות קבלה", "עלות"]},
            {"product": "עזרה בהגשת מועמדות", "concerns": ["תהליך", "עלות", "מועד אחרון"]},
        ],
    },
    {
        "domain": "equipment_rental",
        "domain_he": "השכרת ציוד",
        "scenarios": [
            {"product": "השכרת ציוד לאירוע", "concerns": ["זמינות", "עלות יומית", "הובלה"]},
            {"product": "השכרת כלי עבודה", "concerns": ["יום", "עלות", "פיקדון"]},
        ],
    },
    {
        "domain": "office_maintenance",
        "domain_he": "תחזוקת משרדים",
        "scenarios": [
            {"product": "ניהול תחזוקת בניין", "concerns": ["שירות", "עלות חודשית", "זמינות"]},
            {"product": "תיקונים שוטפים במשרד", "concerns": ["מהירות", "עלות", "אמינות"]},
        ],
    },
    {
        "domain": "tour_guide",
        "domain_he": "מדריכי טיולים",
        "scenarios": [
            {"product": "סיור מודרך בירושלים", "concerns": ["קבוצה", "עלות", "שפה"]},
            {"product": "סיור אוף-רוד", "concerns": ["יעד", "עלות", "ציוד"]},
        ],
    },
]

CUSTOMER_PERSONAS = [
    "משווה מחירים",
    "אימפולסיבי",
    "לקוח חוזר",
    "משפחה צעירה",
    "רגיש למחיר",
    "אנליטי",
    "בעל עסק",
    "רגשי",
    "לקוח יוקרה",
    "חסר סבלנות",
    "שקט",
    "עסוק",
    "ביישן",
    "פטפטן",
    "משא ומתן",
    "מבוגר",
    "צעיר",
    "קריר",
    "הורה",
    "סטודנט",
    "סקפטי",
    "חשדן",
    "ידידותי",
]

BUSINESS_STYLES = ["קצר", "מכירתי", "מקצועי", "רשמי", "חם", "ידידותי", "יעיל"]

WRITING_STYLES = [
    "הודעות קצרות, בלי אמוג'י, שפה יומיומית",
    "הודעות בינוניות, כמה אמוג'י, פסיק אחרי שאלה",
    "הודעות ארוכות, רשמי, פיסוק מלא",
    "הודעות קצרות מאוד, שפת סלנג, אמוג'י",
    "הודעות ללא פיסוק, שפה זורמת",
    "הודעות מפורמטות, ממוספרות, מסודרות",
    "כתיב לא מושלם עם שגיאות קלות",
    "קצר ותכליתי, ללא ניואנסים",
    "חם ואישי, עם פנייה בשם",
    "מקצועי אך נגיש, ניסוח מדויק",
]

TIMING_PATTERNS = [
    "החלפה חיה מהירה עם השהיות של דקות בודדות",
    "שיחה עם פסקות של שעות מספר",
    "המשך ביום שלמחרת",
    "לקוח נעלם ומגיע בחזרה",
    "עסק מעקב אחרי יומיים",
    "פסקת סוף שבוע",
    "פסקה ארוכה לפני דחייה סופית",
    "שיחה חיה עם כמה שאלות ברצף",
]

STARTING_CONTEXTS = [
    "שיחה ראשונה דרך ממליץ",
    "לקוח כבר קיבל הצעה ממתחרה",
    "לקוח דיבר עם ספק אחר בעבר",
    "לקוח ראה פרסום ברשתות חברתיות",
    "לקוח מגיע מאתר של העסק",
    "לקוח מגיע מגוגל",
    "לקוח חוזר שרכש בעבר",
    "לקוח פנה דחוף בגלל תקלה",
    "לקוח פנה בשם מישהו אחר",
    "שיחה ראשונה ללא הקשר",
]

TRAJECTORY_TO_TURNING_POINT = {
    "high_to_conversion": [
        "העסק מסביר בדיוק מה כלול במחיר",
        "לקוח מקבל דוגמה מוצלחת",
        "העסק מציע תאריך מוקדם",
        "לקוח שואל שאלה אחרונה ומרגיש בטוח",
    ],
    "high_to_price_rejection": [
        "לקוח שומע את המחיר ומגיב בשלילה",
        "לקוח מגלה שהמחיר גבוה ממה שציפה",
        "לקוח מציין שמתחרה זול יותר",
    ],
    "high_to_ghosting": [
        "לקוח אומר שישקול ולא חוזר",
        "לקוח מבקש פרטים ולא מגיב",
        "לקוח שולח הודעה אחרונה ונעלם",
    ],
    "high_to_delivery_loss": [
        "לקוח מגלה שהמועד לא מתאים",
        "לקוח מגלה שהמוצר לא זמין בזמן הנדרש",
        "לקוח מאוכזב מזמן האספקה",
    ],
    "high_to_trust_loss": [
        "לקוח מגלה עסק אחר שנתן חוות דעת שלילית",
        "לקוח לא מקבל תשובה בזמן ומתייאש",
        "לקוח חושש מחוסר ניסיון ספציפי",
    ],
    "medium_to_discount_conversion": [
        "העסק מציע חבילה מצומצמת יותר",
        "לקוח שומע על אפשרות ללא התחייבות",
        "העסק מציע הנחה לסגירה מהירה",
    ],
    "medium_to_competitor_loss": [
        "לקוח מציין שמתחרה הציע הרבה פחות",
        "לקוח מציין שקיבל חבילה טובה יותר ממישהו אחר",
    ],
    "medium_to_pending": [
        "לקוח צריך לשאול את השותף",
        "לקוח רוצה לחשוב עוד שבוע",
        "לקוח ממתין לאישור תקציבי",
    ],
    "low_to_appointment": [
        "העסק מציע ייעוץ חינמי ראשוני",
        "לקוח מסכים לשמוע יותר לאחר שהעסק הסביר",
        "לקוח מתרצה ומסכים לפגישה",
    ],
    "low_to_reengagement": [
        "העסק מציע אפשרות ללא התחייבות",
        "לקוח שומע על הטבה מיוחדת ומתעניין מחדש",
        "לקוח חוזר לאחר תקופה עם עניין מחודש",
    ],
}

TRAJECTORY_ENDINGS = {
    "high_to_conversion": "לקוח סוגר עסקה ומתקדם",
    "high_to_price_rejection": "לקוח מסיים את השיחה בסירוב",
    "high_to_ghosting": "לקוח לא מגיב לאחר הבטחה לשוב",
    "high_to_delivery_loss": "לקוח מסיים כי המועד לא מתאים",
    "high_to_trust_loss": "לקוח מאבד אמון ומסיים",
    "medium_to_discount_conversion": "לקוח מסכים לחבילה מוקטנת",
    "medium_to_competitor_loss": "לקוח עובר למתחרה",
    "medium_to_pending": "שיחה מסתיימת ללא החלטה סופית",
    "low_to_appointment": "תור/פגישה נקבעת",
    "low_to_reengagement": "לקוח מחדש עניין ומבקש פרטים",
}

MESSAGE_COUNT_WEIGHTS = {
    7: 25,
    9: 35,
    11: 15,
    13: 5,
    15: 5,
    17: 4,
    19: 3,
    21: 4,
    25: 2,
    29: 1,
    33: 1,
}

AGE_IMPRESSIONS = ["צעיר (20-30)", "ביניים (30-45)", "בגיל מבוגר (50+)", "לא ברור"]
TECHNICAL_LEVELS = ["לא טכני", "בסיסי", "בינוני", "טכני"]
URGENCY_LEVELS = ["דחוף מאוד", "בינוני", "לא דחוף", "יכול להמתין"]
BUDGET_SENSITIVITIES = ["מאוד רגיש למחיר", "רגיש", "גמיש", "לא רגיש למחיר"]
TRUST_LEVELS = ["אפס אמון", "חשדן", "סביר", "אמין", "מאמין"]
OBJECTION_TYPES = {
    "high_to_price_rejection": ["מחיר גבוה מדי", "הצעת מחיר לא תחרותית"],
    "high_to_delivery_loss": ["מועד אספקה לא מתאים", "זמן ביצוע ארוך מדי"],
    "high_to_trust_loss": ["חוסר אמון בעסק", "חוסר ניסיון", "ביקורות שליליות"],
    "medium_to_competitor_loss": ["מתחרה עם הצעה טובה יותר", "מחיר נמוך יותר אצל מתחרה"],
    "high_to_ghosting": ["לא מסביר", "לקוח נעלם ללא סיבה ברורה"],
    "medium_to_pending": ["צריך לשאול שותף/בן זוג/מנהל", "חוסר החלטיות"],
    "high_to_conversion": ["ללא התנגדות משמעותית"],
    "medium_to_discount_conversion": ["מחיר גבוה מהתקציב", "בקשה לחבילה קטנה יותר"],
    "low_to_appointment": ["התנגדות ראשונית", "ספקנות"],
    "low_to_reengagement": ["חוסר עניין ראשוני", "מחיר"],
}


def _weighted_choice(weights: dict[int, int], rng: random.Random) -> int:
    items = list(weights.keys())
    w = list(weights.values())
    return rng.choices(items, weights=w, k=1)[0]


def _choose_underrepresented(distribution: dict[str, int], options: list[str], rng: random.Random) -> str:
    if not distribution:
        return rng.choice(options)
    min_count = min(distribution.get(o, 0) for o in options)
    candidates = [o for o in options if distribution.get(o, 0) <= min_count + 2]
    return rng.choice(candidates) if candidates else rng.choice(options)


def create_conversation_plan(
    next_id: str,
    rng: random.Random,
    existing_domain_dist: dict[str, int] | None = None,
    existing_trajectory_dist: dict[str, int] | None = None,
    used_plan_signatures: set[str] | None = None,
) -> ConversationPlan:
    existing_domain_dist = existing_domain_dist or {}
    existing_trajectory_dist = existing_trajectory_dist or {}
    used_plan_signatures = used_plan_signatures or set()

    all_trajectories = list(TRAJECTORY_TO_TURNING_POINT.keys())
    trajectory = _choose_underrepresented(existing_trajectory_dist, all_trajectories, rng)

    all_domains = [d["domain"] for d in DOMAIN_CATALOG]
    domain_entry = None
    for attempt in range(20):
        candidate_domain = _choose_underrepresented(existing_domain_dist, all_domains, rng)
        entry = next(d for d in DOMAIN_CATALOG if d["domain"] == candidate_domain)
        scenario = rng.choice(entry["scenarios"])
        sig = f"{candidate_domain}:{scenario['product']}:{trajectory}"
        if sig not in used_plan_signatures:
            domain_entry = entry
            break
    if domain_entry is None:
        domain_entry = rng.choice(DOMAIN_CATALOG)
        scenario = rng.choice(domain_entry["scenarios"])

    objection_options = OBJECTION_TYPES.get(trajectory, ["כללי"])
    objection_type = rng.choice(objection_options)

    turning_points = TRAJECTORY_TO_TURNING_POINT[trajectory]
    turning_point = rng.choice(turning_points)
    ending = TRAJECTORY_ENDINGS[trajectory]

    message_count = _weighted_choice(MESSAGE_COUNT_WEIGHTS, rng)
    if trajectory == "medium_to_competitor_loss" and rng.random() < 0.3:
        message_count = 8

    customer_persona = rng.choice(CUSTOMER_PERSONAS)
    business_style = rng.choice(BUSINESS_STYLES)
    gender = rng.choice(["male", "female"])
    age_impression = rng.choice(AGE_IMPRESSIONS)
    technical_level = rng.choice(TECHNICAL_LEVELS)
    writing_style = rng.choice(WRITING_STYLES)
    timing_pattern = rng.choice(TIMING_PATTERNS)
    urgency = rng.choice(URGENCY_LEVELS)
    budget_sensitivity = rng.choice(BUDGET_SENSITIVITIES)
    trust_level = rng.choice(TRUST_LEVELS)
    starting_context = rng.choice(STARTING_CONTEXTS)

    concern_list = scenario.get("concerns", ["עלות", "איכות"])
    main_concern = rng.choice(concern_list)
    remaining = [c for c in concern_list if c != main_concern]
    secondary_concern = rng.choice(remaining) if remaining else "שירות"

    product = scenario["product"]
    specific_scenarios = [
        f"לקוח מחפש {product} ב{domain_entry['domain_he']}",
        f"לקוח זקוק ל{product} לבית",
        f"עסק מחפש {product}",
        f"לקוח שמע עליכם ורוצה לברר על {product}",
    ]
    specific_scenario = rng.choice(specific_scenarios)

    customer_goals = [
        f"להבין אם {product} מתאים לצרכים",
        f"לקבל מחיר ל{product}",
        f"לקבוע פגישה לבחינת {product}",
        f"להשוות הצעות עבור {product}",
    ]
    customer_goal = rng.choice(customer_goals)

    return ConversationPlan(
        conversation_id=next_id,
        domain=domain_entry["domain"],
        domain_he=domain_entry["domain_he"],
        specific_scenario=specific_scenario,
        product_or_service=product,
        customer_gender=gender,
        customer_persona=customer_persona,
        customer_age_impression=age_impression,
        customer_technical_level=technical_level,
        business_style=business_style,
        customer_goal=customer_goal,
        main_concern=main_concern,
        secondary_concern=secondary_concern,
        starting_context=starting_context,
        trajectory=trajectory,
        turning_point=turning_point,
        ending=ending,
        message_count=message_count,
        writing_style=writing_style,
        timing_pattern=timing_pattern,
        urgency=urgency,
        budget_sensitivity=budget_sensitivity,
        trust_level=trust_level,
        objection_type=objection_type,
    )
