# -----------------------------------------------------------------------------
# Projekt: Automatische Klassifizierung von Support-Tickets (Interaktive Version)
# Ansatz:  Finale, intelligente Drei-Stufen-Pipeline mit spezialisierten Modellen
# -----------------------------------------------------------------------------

from transformers import pipeline

# --- KONFIGURATION (Version 13: Intelligente Pipeline) ---

# MODELL 1: Der Spezialist für "Gefühle" (Emotion-Scanner)
SENTIMENT_MODELL = "oliverguhr/german-sentiment-bert"

# MODELL 2 & 3: Der flexible Allrounder für Absicht und Thema
ZEROSHOT_MODELL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# --- KATEGORIEN-DATENBANK ---
# Unsere klaren, finalen Labels
KATEGORIEN = {
    "Technischer Fehler: Absturz": "Das Programm ist unbenutzbar (stürzt ab, friert ein).",
    "Technischer Fehler: Funktion": "Ein spezifischer Teil des Programms verhält sich unerwartet.",
    "Anleitung & Bedienungsfrage": "Der Nutzer benötigt Wissen oder eine Anleitung (z.B. 'Wie mache ich...?').",
    "Rechnung & Finanzen": "Anfragen zu Geld, Bezahlung, Laufzeiten oder Kauf (Rechnung, Abo, Lizenz).",
    "Login & Passwort Problem": "Der Nutzer kommt nicht in seinen Account (Login, Passwort vergessen, gesperrt).",
    "Profil & Datenverwaltung": "Der Nutzer ist eingeloggt und möchte seine Daten ändern (Name, Adresse, E-Mail).",
    "Positives Feedback & Lob": "Der Nutzer ist glücklich und möchte seine Zufriedenheit ausdrücken.",
    "Kritik & Verbesserungsvorschlag": "Der Nutzer ist unzufrieden mit dem Design oder hat eine Idee für eine neue Funktion.",
    "Kritik an Support & Personal": "Der Nutzer ist unzufrieden mit einer Person, einem Prozess oder der Servicequalität.",
    "Spam & Irrelevante Anfrage": "Die Anfrage hat nichts mit dem Produkt zu tun (Werbung, Bewerbung)."
}

# --- FUNKTIONEN ---
def initialisiere_pipelines():
    print(f"Lade Sentiment-Modell (Stufe 1) '{SENTIMENT_MODELL}'...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODELL)
        print("Sentiment-Modell erfolgreich geladen!")
    except Exception as e:
        print(f"FEHLER beim Laden des Sentiment-Modells: {e}")
        return None, None

    print(f"\nLade Zero-Shot-Modell (Stufe 2 & 3) '{ZEROSHOT_MODELL}'...")
    try:
        zeroshot_pipeline = pipeline("zero-shot-classification", model=ZEROSHOT_MODELL)
        print("Zero-Shot-Modell erfolgreich geladen!")
    except Exception as e:
        print(f"FEHLER beim Laden des Zero-Shot-Modells: {e}")
        return None, None
        
    print("\nAlle Modelle geladen! Das Programm ist jetzt startklar.\n")
    return sentiment_pipeline, zeroshot_pipeline

# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    sentiment_classifier, themen_classifier = initialisiere_pipelines()

    if sentiment_classifier and themen_classifier:
        print("="*60)
        print("    Interaktiver Support-Ticket-Klassifikator (V2.0 - Intelligente Pipeline)")
        print("="*60)
        print("Geben Sie eine Support-Anfrage ein oder 'exit' zum Beenden.")
        print("-" * 60)

        while True:
            user_input = input("\nIhre Anfrage: ")
            if user_input.lower() == 'exit':
                print("Programm wird beendet. Auf Wiedersehen!")
                break

            # === STUFE 1: Emotion-Scanner ===
            sentiment_ergebnis = sentiment_classifier(user_input)[0]
            stimmung = sentiment_ergebnis['label']
            print(f"   > Stufe 1 (Emotion): '{stimmung.upper()}'")

            # === STUFE 2: Absicht-Detektiv ===
            absicht_labels = ["Eine Frage stellen", "Ein Problem melden", "Eine Meinung oder Feedback geben"]
            absicht_ergebnis = themen_classifier(user_input, absicht_labels, multi_label=False)
            absicht = absicht_ergebnis['labels'][0]
            print(f"   > Stufe 2 (Absicht): '{absicht}'")

            # === STUFE 3: Intelligente Themen-Auswahl basierend auf Stufe 1 & 2 ===
            finale_kategorien = []
            if stimmung == 'positive':
                finale_kategorien = ["Positives Feedback & Lob"]
            elif absicht == "Eine Frage stellen" and stimmung != 'negative':
                finale_kategorien = ["Anleitung & Bedienungsfrage", "Rechnung & Finanzen", "Profil & Datenverwaltung"]
            elif absicht == "Ein Problem melden":
                finale_kategorien = ["Technischer Fehler: Absturz", "Technischer Fehler: Funktion", "Login & Passwort Problem"]
            elif stimmung == 'negative': # Alles andere, das negativ ist, ist Kritik
                finale_kategorien = ["Kritik & Verbesserungsvorschlag", "Kritik an Support & Personal"]
            else: # Fallback für alles andere
                finale_kategorien = list(KATEGORIEN.keys())

            # === Finale Klassifizierung mit dem Themen-Experten ===
            if finale_kategorien:
                finales_ergebnis = themen_classifier(user_input, finale_kategorien, multi_label=False)
                bester_label = finales_ergebnis['labels'][0]
                bester_score = finales_ergebnis['scores'][0]
                print(f"   > Stufe 3 (Thema): '{bester_label}' (Sicherheit: {bester_score:.2%})")
            else:
                print("   > Stufe 3 (Thema): Konnte keine passenden Kategorien finden.")