"""
Support-Ticket-Klassifikator (V3) - Verbesserte, robuste und erklärbare Pipeline

Hauptziele der Überarbeitung:
- Robusterer Umgang mit Modell-Labels (Groß-/Kleinschreibung, verschiedene Rückgabeformate)
- Kombination von Zero-Shot mit einfachen Keyword-Regeln für bessere Präzision
- Multi-Label-Ausgaben nutzen und Confidence-Schwellen für Entscheidungen
- Klare Ausgabe, Unsicherheits-Markierung und Fallbacks
- Modularer Aufbau (einfacher Test / Erweiterung)

Voraussetzungen:
pip install transformers[torch]==4.35.0 torch --upgrade
(oder: pip install -r requirements.txt)

Hinweis: Zero-Shot-Modelle können recht groß sein; falls Du Speicherprobleme hast,
nutze eine kleinere oder quantisierte Alternative.
"""

from transformers import pipeline
import re
import sys

# ---------------- KONFIGURATION ----------------
SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert" 
ZEROSHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Finale Kategorien (deutsche Labels)
KATEGORIEN = {
    "Technischer Fehler: Absturz": "Das Programm ist unbenutzbar (stürzt ab, friert ein).",
    "Technischer Fehler: Funktion": "Ein spezifischer Teil des Programms verhält sich unerwartet.",
    "Anleitung & Bedienungsfrage": "Der Nutzer benötigt Wissen oder eine Anleitung (z.B. 'Wie mache ich...?').",
    "Rechnung & Finanzen": "Anfragen zu Geld, Bezahlung, Rechnungen, Laufzeiten oder Kauf (Rechnung, Abo, Lizenz).",
    "Login & Passwort Problem": "Der Nutzer kommt nicht in seinen Account (Login, Passwort vergessen, gesperrt).",
    "Profil & Datenverwaltung": "Der Nutzer ist eingeloggt und möchte seine Daten ändern (Name, Adresse, E-Mail).",
    "Positives Feedback & Lob": "Der Nutzer ist glücklich und möchte seine Zufriedenheit ausdrücken.",
    "Kritik & Verbesserungsvorschlag": "Der Nutzer ist unzufrieden mit dem Design oder hat eine Idee für eine neue Funktion.",
    "Kritik an Support & Personal": "Der Nutzer ist unzufrieden mit einer Person, einem Prozess oder der Servicequalität.",
    "Spam & Irrelevante Anfrage": "Die Anfrage hat nichts mit dem Produkt zu tun (Werbung, Bewerbung)."
}

# Keyword-basiertes Mapping (einfacher Heuristik-Boost)
KEYWORD_MAP = {
    r"\brechnung\b|\babo\b|\bmonatlich\b|\bzahlung\b|\bkosten\b": "Rechnung & Finanzen",
    r"\bpasswort\b|\blogin\b|\banmelden\b|\bkonto\b|\bgesperrt\b|\bzugang\b": "Login & Passwort Problem",
    r"\babsturz\b|\bcrash\b|\bfriert\b|\b(stürzt|abstürzt)\b|\bneustart\b": "Technischer Fehler: Absturz",
    r"\bfehler\b|\bbug\b|\bfunktioniert nicht\b|\bfunktioniert\b|\bgeht nicht\b|\bproble?m\b": "Technischer Fehler: Funktion",
    r"\banleitung\b|\bbedienung\b|\bwie\b|\bhilfe\b|\bschritt\b": "Anleitung & Bedienungsfrage",
    r"\bprofil\b|\badresse\b|\be-mail\b|\bemail\b|\bname\b|\bdaten\b": "Profil & Datenverwaltung",
    r"\blob\b|\bdanke\b|\bsuper\b|\bgefällt\b|\bzufrieden\b": "Positives Feedback & Lob",
    r"\bkritik\b|\bschlecht\b|\bverbesser\b|\bhass\b|\bnicht gut\b": "Kritik & Verbesserungsvorschlag",
    r"\bsupport\b|\bmitarbeiter\b|\bangestellter\b|\bbeleidigt\b": "Kritik an Support & Personal",
    r"\bwerbung\b|\bspam\b|\bbewerbung\b|\bangebot\b": "Spam & Irrelevante Anfrage"
}

# Schwellenwerte
INTENT_CONFIDENCE_THRESHOLD = 0.45  # unterhalb dessen markiert die Pipeline als unsicher
FINAL_CONFIDENCE_THRESHOLD = 0.50

# -----------------------------------------------

def load_models(device=None):
    """Lädt die benötigten Pipelines. device=None lässt transformers entscheiden (CPU/GPU).
    Gib device=0 für GPU, -1 für CPU."""
    print(f"Lade Sentiment-Modell '{SENTIMENT_MODEL}'...")
    sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)
    print("Sentiment geladen.")

    print(f"Lade Zero-Shot-Modell '{ZEROSHOT_MODEL}' (für Intent & Thema)...")
    zeroshot_pipe = pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=device)
    print("Zero-Shot geladen.")

    return sentiment_pipe, zeroshot_pipe


def normalize_label(label: str) -> str:
    return label.strip().lower()


def keyword_guess(text: str):
    text_l = text.lower()
    for pattern, category in KEYWORD_MAP.items():
        if re.search(pattern, text_l):
            return category
    return None


def detect_sentiment(pipe, text: str):
    res = pipe(text)[0]
    # robust gegen verschiedene Label-Formate
    label = normalize_label(res.get('label', ''))
    score = float(res.get('score', 0.0))
    # mögliche Varianten: 'POSITIVE', 'Negative', 'LABEL_0' (selten)
    if label.startswith('pos'):
        label = 'positive'
    elif label.startswith('neg'):
        label = 'negative'
    else:
        label = 'neutral'
    return label, score


def detect_intent(pipe, text: str, candidate_labels, multi_label=True):
    # Formuliere Hypothese auf Deutsch - verbessert die Genauigkeit bei multilingualen MNLIs
    hypothesis_template = "Diese Nachricht handelt von {}."
    res = pipe(text, candidate_labels, hypothesis_template=hypothesis_template, multi_label=multi_label)
    return res


def assemble_candidate_categories(sentiment_label, intent_top_labels):
    # Basierend auf Absichtsvorhersage und Stimmung eine engere Auswahl an Kategorien erzeugen
    candidates = []

    # Direkter Boost: positives Feedback
    if sentiment_label == 'positive':
        candidates.append('Positives Feedback & Lob')

    # Intent-basierte Zuordnung (labels sind deutsche Kurzabsichten)
    # Wir prüfen Schlüsselwörter in den intent_top_labels
    for lab in intent_top_labels:
        lab_l = lab.lower()
        if 'problem' in lab_l or 'melden' in lab_l:
            candidates.extend([
                'Technischer Fehler: Absturz',
                'Technischer Fehler: Funktion',
                'Login & Passwort Problem'
            ])
        if 'frage' in lab_l:
            candidates.extend([
                'Anleitung & Bedienungsfrage',
                'Rechnung & Finanzen',
                'Profil & Datenverwaltung'
            ])
        if 'feedback' in lab_l or 'meinung' in lab_l:
            candidates.extend([
                'Kritik & Verbesserungsvorschlag',
                'Kritik an Support & Personal'
            ])

    # Fallback: falls leer -> alle Kategorien
    if not candidates:
        candidates = list(KATEGORIEN.keys())

    # Entferne Duplikate, erhalte Reihenfolge
    candidates = list(dict.fromkeys(candidates))
    return candidates


def final_decision(zeroshot_pipe, text: str, candidates, keyword_override=None):
    # Wenn ein Keyword-Match gefunden wurde, berücksichtigen wir das als starken Hinweis
    if keyword_override and keyword_override in candidates:
        # Wir prüfen trotzdem das Kandidaten-Scoring und bevorzugen Keyword, wenn kein klarer Sieger
        result = detect_intent(zeroshot_pipe, text, candidates, multi_label=True)
        # Wenn keyword schon den höchsten Score hat, return
        labels = result.get('labels', [])
        scores = result.get('scores', [])
        if labels:
            top_label = labels[0]
            top_score = scores[0]
            if top_label == keyword_override or top_score < FINAL_CONFIDENCE_THRESHOLD:
                return keyword_override, 0.0, True
        return top_label, top_score, False

    # Normaler Ablauf: Zero-shot über die Kandidaten (multi_label -> wir nehmen bestes)
    result = detect_intent(zeroshot_pipe, text, candidates, multi_label=True)
    labels = result.get('labels', [])
    scores = result.get('scores', [])
    if not labels:
        return None, 0.0, False
    return labels[0], scores[0], False


# ----------------- Interaktive Schleife -----------------
if __name__ == '__main__':
    try:
        sentiment_pipe, zeroshot_pipe = load_models()
    except Exception as e:
        print("Fehler beim Laden der Modelle:", e, file=sys.stderr)
        sys.exit(1)

    print("\nInteraktiver Support-Ticket-Klassifikator (V3)")
    print("Tip: 'exit' zum Beenden\n")

    while True:
        text = input("Ihre Anfrage: ").strip()
        if not text:
            continue
        if text.lower() == 'exit':
            print("Beende Programm. Auf Wiedersehen!")
            break

        # 1) Keyword-Boost (schnell, deterministisch)
        kw = keyword_guess(text)
        if kw:
            print(f"   [Heuristik] Gefundene Stichworte -> Vorschlag: {kw}")

        # 2) Sentiment-Analyse
        sentiment_label, sentiment_score = detect_sentiment(sentiment_pipe, text)
        print(f"   > Stufe 1 (Emotion): '{sentiment_label.upper()}' (score={sentiment_score:.2%})")

        # 3) Absicht (Intent) - engere Labels auf Deutsch
        absicht_labels = ["Eine Frage stellen", "Ein Problem melden", "Eine Meinung oder Feedback geben", "Sonstiges"]
        absicht_result = detect_intent(zeroshot_pipe, text, absicht_labels, multi_label=True)
        # Zeige die Top-3 Absichten
        print("   > Stufe 2 (Absicht) - Top-Vorhersagen:")
        for lab, sc in zip(absicht_result.get('labels', [])[:3], absicht_result.get('scores', [])[:3]):
            print(f"       - {lab} ({sc:.2%})")

        # 4) Kandidaten zusammenstellen
        top_intents = absicht_result.get('labels', [])[:2]
        candidates = assemble_candidate_categories(sentiment_label, top_intents)
        print(f"   > Stufe 3 (Vorauswahl): {len(candidates)} Kandidaten")

        # 5) Finale Entscheidung
        final_label, final_score, used_keyword = final_decision(zeroshot_pipe, text, candidates, keyword_override=kw)

        # 6) Ausgabe mit Unsicherheits-Markierung
        if used_keyword:
            print(f"--> FINALES ERGEBNIS (Keyword-Override): '{final_label}' (unspezifiziertes Score)")
        elif final_score < FINAL_CONFIDENCE_THRESHOLD:
            print(f"--> FINALES ERGEBNIS (unsicher): '{final_label}' (Sicherheit: {final_score:.2%})")
            # zusätzlich Top-3 anbieten
            alt_labels = detect_intent(zeroshot_pipe, text, list(KATEGORIEN.keys()), multi_label=True)
            print("    Alternative Vorschläge:")
            for lab, sc in zip(alt_labels.get('labels', [])[:3], alt_labels.get('scores', [])[:3]):
                print(f"       - {lab} ({sc:.2%})")
        else:
            print(f"--> FINALES ERGEBNIS: '{final_label}' (Sicherheit: {final_score:.2%})")

        print("-"*60)