# -----------------------------------------------------------------------------
# Projekt: Automatische Klassifizierung von Support-Tickets (Interaktive Version)
# Modell:  MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
# Zweck:   Demonstration mit verbesserten, beschreibenden Kategorien
# -----------------------------------------------------------------------------

from transformers import pipeline

# --- KONFIGURATION (mit verbesserten Kategorien) ---
TICKET_KATEGORIEN = [
    "Technischer Fehler / Bug (Etwas funktioniert nicht wie erwartet, z.B. Absturz, Fehlermeldung, Button ohne Funktion)",
    "Frage zur Bedienung / Feature (Anfrage, wie man eine bestimmte Funktion nutzt oder ob es ein Feature gibt)",
    "Rechnung & Zahlung (Fragen zu Rechnungsbeträgen, Zahlungsstatus, Abonnements)",
    "Login & Passwort (Probleme beim Anmelden, Passwort vergessen oder zurücksetzen)",
    "Stammdaten & Profil (Benutzername, E-Mail-Adresse oder persönliche Informationen ändern)",
    "Feedback & Vorschläge (Lob, Kritik oder Ideen für neue Funktionen)",
    "Sonstige Anfrage (Das Anliegen passt in keine der anderen Kategorien)"
]
MODELL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# --- FUNKTIONEN ---
def initialisiere_klassifikator():
    print(f"Lade das Klassifizierungsmodell '{MODELL_NAME}'...")
    try:
        classifier = pipeline("zero-shot-classification", model=MODELL_NAME)
        print("Modell erfolgreich geladen! Das Programm ist jetzt startklar.\n")
        return classifier
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return None

def klassifiziere_text(classifier, text, kategorien):
    if not classifier:
        return None
    return classifier(text, kategorien, multi_label=False)

# --- HAUPTPROGRAMM (INTERAKTIVE SCHLEIFE) ---
if __name__ == "__main__":
    ticket_classifier = initialisiere_klassifikator()

    if ticket_classifier:
        print("="*60)
        print("    Interaktiver Support-Ticket-Klassifikator (Version 1.1)")
        print("="*60)
        print("Geben Sie eine Support-Anfrage ein, um sie zu klassifizieren.")
        print("Geben Sie 'exit' ein, um das Programm zu beenden.")
        print("-" * 60)

        while True:
            user_input = input("\nIhre Anfrage: ")

            if user_input.lower() == 'exit':
                print("Programm wird beendet. Auf Wiedersehen!")
                break
            
            ergebnis = klassifiziere_text(ticket_classifier, user_input, TICKET_KATEGORIEN)
            
            if ergebnis:
                beste_kategorie = ergebnis['labels'][0]
                vertrauensscore = ergebnis['scores'][0]
                
                print(f"-> KI-Analyse: Diese Anfrage gehört am wahrscheinlichsten zur Kategorie '{beste_kategorie}' (Sicherheit: {vertrauensscore:.2%}).")
            else:
                print("-> Klassifikation fehlgeschlagen.")