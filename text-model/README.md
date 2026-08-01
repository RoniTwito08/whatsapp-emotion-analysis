# text-model

## מטרת המודל

תיקייה זו מכינה את צינור הקלט (input pipeline) עבור מודל הטקסט המרכזי של הפרויקט —
מודל Transformer מבוסס עברית שיסווג שיחות לקוח (WhatsApp) לפי רמת עניין: `interested`
מול `losing_interest`. **המשימה הנוכחית היא הכנה בלבד.** לא מתבצע כאן שום אימון של
המודל, ולא מיוצרות תוצאות אימון או מדדי ביצועים (accuracy / F1) — אלה יגיעו בשלב הבא,
לאחר שהקורפוס הסופי יהיה מוכן.

## למה AlephBERT?

`onlplab/alephbert-base` הוא מודל BERT שאומן מראש על טקסט עברי, ולכן הוא נקודת התחלה
טבעית למשימת סיווג טקסט בעברית (במקום מודל אנגלי או רב-לשוני כללי). מכיוון שזהו
צ'קפוינט (checkpoint) בסיסי (base) בלבד, **אין בו ראש סיווג (classification head)**
מותאם למשימה שלנו — ראו הסבר על כך בהמשך.

## מבנה הפרויקט

```
text-model/
├── config.json          # כל ההגדרות הניתנות לשינוי (מודל, שדות, טוקנייזר, דאטהלואדר)
├── dataset.py            # HebrewConversationDataset - טעינה, מיפוי תוויות, טוקניזציה
├── data_loader.py         # טעינת קונפיג, טוקנייזר, מודל, Dataset ו-DataLoader
├── validate_inputs.py     # בדיקת עשן (smoke test) מקצה לקצה כולל forward pass
├── inspect_batch.py       # הצגת דוגמת batch מטוקנז לבדיקה ידנית
├── requirements.txt
├── README.md              # הקובץ הזה
├── data/
│   ├── .gitkeep
│   └── sample_corpus.jsonl   # קורפוס דמה סינתטי, לבדיקה טכנית בלבד
└── outputs/
    └── .gitkeep               # ליציאות עתידיות (לא נשמר ב-git)
```

## הקמת סביבה וירטואלית ב-Windows

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

⚠️ ההורדה הראשונה של PyTorch ושל המודל `onlplab/alephbert-base` מ-Hugging Face עלולה
להיות **כבדה (מאות MB) ולקחת זמן**, בהתאם למהירות האינטרנט. ריצות הבאות ישתמשו במטמון
המקומי (`~/.cache/huggingface`) ויהיו מהירות בהרבה.

## היכן למקם את הקורפוס הסופי

כאשר הקורפוס הסופי (אמיתי, לא סינתטי) יהיה מוכן, יש להניח אותו בתיקיית `data/`
(לדוגמה: `data/final_corpus.jsonl`).

## עדכון config.json עם הגעת הקורפוס

יש לעדכן בקובץ `config.json` את השדות הבאים תחת `data`:

- `input_path` — הנתיב היחסי לקובץ הקורפוס החדש (לדוגמה `"data/final_corpus.jsonl"`).
- `input_format` — `"jsonl"`, `"json"` או `"csv"`, בהתאם לפורמט הקובץ בפועל.
- `id_field`, `messages_field`, `role_field`, `text_field`, `label_field` — אם שמות
  השדות בקורפוס האמיתי שונים משמות השדות בדוגמה, יש לעדכן אותם כאן.
- `included_roles` — אילו תפקידים (roles) בשיחה ייכללו בטקסט (כרגע רק `"customer"`).
- `label_mapping` — אילו תוויות גולמיות (raw labels) ממופות לכל אחת משתי המחלקות
  הסופיות.

שאר הקוד (`dataset.py`, `data_loader.py`) קורא את כל הערכים הללו מתוך `config.json`
ואינו זקוק לשינוי כאשר משנים קורפוס.

## איך עובד מיפוי התוויות (label mapping)

כל שיחה מגיעה עם תווית גולמית בשדה `final_outcome` (למשל `converted`, `ghosted`).
`label_mapping` בקונפיג ממפה כל תווית גולמית כזו לאחת משתי המחלקות הסופיות:

```
"interested":       ["converted", "appointment_set", "pending", "reengaged_pending"]
"losing_interest":  ["explicit_rejection", "competitor_loss", "delivery_loss",
                      "trust_loss", "ghosted"]
```

רשומות עם תווית גולמית שלא מופיעה בשום רשימה — **מדולגות (ignored)** ולא נכנסות
ל-Dataset. מזהי המחלקות הסופיות (`label_to_id`) נקבעים באופן דטרמיניסטי (מיון
אלפביתי של שמות המחלקות), כך שעבור המשימה הבינארית ברירת המחדל מתקבל תמיד:

```
{"interested": 0, "losing_interest": 1}
```

## איך מחוברות הודעות הלקוח

מתוך רשימת ה-`messages` של כל שיחה, נשמרות רק ההודעות ששדה ה-`role` שלהן נמצא
ברשימת `included_roles` (כרגע: `customer` בלבד). ההודעות הנשמרות מחוברות למחרוזת
טקסט אחת בעזרת `message_separator` (ברירת מחדל: `" [SEP] "`).

## max_length, padding ו-truncation

ההגדרות תחת `tokenizer` בקונפיג שולטות על הטוקניזציה:

- `max_length` — אורך הרצף המקסימלי (בטוקנים) שהמודל יקבל.
- `padding: "max_length"` — כל הרצפים מרופדים (padded) לאותו אורך קבוע, כדי
  שאפשר יהיה לאגד (batch) אותם לטנזור אחיד.
- `truncation: true` — טקסטים ארוכים מ-`max_length` נחתכים.

טוקניזציה מתבצעת פעם אחת בלבד, בזמן יצירת ה-`HebrewConversationDataset`, ולא בכל
epoch מחדש.

## הרצת validate_inputs.py

```
python validate_inputs.py --config config.json
```

הסקריפט מבצע בדיקת עשן (smoke test) מלאה: טוען קונפיג, טוקנייזר, Dataset,
DataLoader ומודל, שולף batch אחד, מריץ עליו את כל בדיקות התקינות (dtypes, shapes,
טווחי ערכים, NaN/Inf וכו'), מריץ forward pass יחיד תחת `model.eval()` +
`torch.no_grad()`, ומדפיס דוח תקינות. קוד היציאה הוא שונה מ-0 אם משהו נכשל.

## הרצת inspect_batch.py

```
python inspect_batch.py --config config.json
```

הסקריפט מדפיס לכל רשומה ב-batch: מזהה רשומה, הטקסט המחובר, שם ומספר התווית,
מספר הטוקנים לפני ריפוד (padding), 30 מזהי הטוקנים הראשונים, 30 הטוקנים הראשונים,
וטקסט מפוענח (decoded). זהו כלי לבדיקה ידנית שהטוקניזציה בעברית נראית סבירה.

## אזהרת "newly initialized" של ראש הסיווג

בזמן טעינת המודל תוצג (בסבירות גבוהה) אזהרה דומה ל:

```
Some weights of BertForSequenceClassification were not initialized from the
model checkpoint ... and are newly initialized: ['classifier.weight', ...]
You should probably TRAIN this model on a down-stream task...
```

**זו אזהרה צפויה ולא שגיאה.** הצ'קפוינט הבסיסי `onlplab/alephbert-base` לא כולל
ראש סיווג למשימה שלנו (סיווג בינארי interested / losing_interest), כך ש-
Transformers יוצר ראש סיווג חדש ואקראי במקומו. ראש זה יאומן בשלב האימון העתידי —
כרגע הוא רק חלק מבדיקת התקינות של הצינור.

## בחירת CUDA / CPU

הקוד בודק אוטומטית `torch.cuda.is_available()`: אם קיים GPU תואם CUDA, המודל
עובר אליו; אחרת הוא נשאר על ה-CPU. המכשיר הנבחר (`cuda` או `cpu`) מודפס בכל
הרצה של `data_loader.load_model`.

## לגבי מדדי ביצועים

מכיוון שבשלב הזה **אין אימון בכלל**, לא מיוצרים (ולא צריכים להיות מיוצרים) שום
מדדי accuracy, F1, precision/recall או תוצאות אימון אחרות. `validate_inputs.py`
מדווח רק loss בודד מ-forward pass יחיד לצורך בדיקת תקינות — זה **אינו** מדד ביצועים
של מודל מאומן.

## קורפוס הדוגמה (sample_corpus.jsonl)

הקובץ `data/sample_corpus.jsonl` מכיל 8 שיחות עבריות **סינתטיות** קצרות
(מסומנות `"synthetic": true`), 4 ממופות ל-`interested` ו-4 ל-`losing_interest`.
מטרתו היחידה היא לבדוק טכנית את ה-Dataset, הטוקנייזר, ה-DataLoader ואת קלטי
המודל — **הוא אינו נתוני מחקר או קורפוס אמיתי**, ואין להשתמש בו לאימון בפועל.
