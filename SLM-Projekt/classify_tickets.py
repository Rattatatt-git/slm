# -----------------------------------------------------------------------------
# Projekt: Automatische Klassifizierung von Support-Tickets (Interaktive Version)
# Modell:  MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
# Zweck:   Finale, funktionierende Version mit klaren, eindeutigen Labels
# -----------------------------------------------------------------------------

from transformers import pipeline

# --- KONFIGURATION (Version 9: Finale, klare Labels) ---
KATEGORIEN_DETAILS = {
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

TICKET_LABELS = list(KATEGORIEN_DETAILS.keys())
MODELL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# --- FUNKTIONEN --- (unverändert)
def initialisiere_klassifikator():
    print(f"Lade das Klassifizierungsmodell '{MODELL_NAME}'...")
    try:
        classifier = pipeline("zero-shot-classification", model=MODELL_NAME)
        print("Modell erfolgreich geladen! Das Programm ist jetzt startklar.\n")
        return classifier
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return None

def klassifiziere_text(classifier, text, labels):
    if not classifier:
        return None
    return classifier(text, labels, multi_label=False)

# --- HAUPTPROGRAMM --- (unverändert)
if __name__ == "__main__":
    ticket_classifier = initialisiere_klassifikator()

    if ticket_classifier:
        print("="*60)
        print("    Interaktiver Support-Ticket-Klassifikator (Version 6.0 - Final)")
        print("="*60)
        print("Geben Sie eine Support-Anfrage ein oder 'exit' zum Beenden.")
        print("-" * 60)

        while True:
            user_input = input("\nIhre Anfrage: ")
            if user_input.lower() == 'exit':
                print("Programm wird beendet. Auf Wiedersehen!")
                break
            
            ergebnis = klassifiziere_text(ticket_classifier, user_input, TICKET_LABELS)
            
            if ergebnis:
                bester_label = ergebnis['labels'][0]
                bester_score = ergebnis['scores'][0]

                if bester_score >= 0.60: # Wir können wieder selbstbewusster sein
                    print(f"-> KI-Analyse (Sicher): Diese Anfrage gehört zur Kategorie '{bester_label}' (Sicherheit: {bester_score:.2%}).")
                else:
                    zweitbester_label = ergebnis['labels'][1]
                    zweitbester_score = ergebnis['scores'][1]
                    print(f"-> KI-Analyse (Unsicher): Das Anliegen konnte nicht eindeutig zugeordnet werden.")
                    print(f"   Top-Vermutung:    '{bester_label}' (mit {bester_score:.2%})")
                    print(f"   Zweite Vermutung: '{zweitbester_label}' (mit {zweitbester_score:.2%})")
                    print(f"   --> Diese Anfrage sollte manuell geprüft werden.")
            else:
                print("-> Klassifikation fehlgeschlagen.")