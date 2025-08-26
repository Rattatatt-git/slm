# -----------------------------------------------------------------------------
# Projekt: Automatische Klassifizierung von Support-Tickets (Interaktive Version)
# Modell:  MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (Zero-Shot Classification)
# Zweck:   Interaktive Demonstration für die Projektaufgabe
# -----------------------------------------------------------------------------

from transformers import pipeline

# --- KONFIGURATION ---
TICKET_KATEGORIEN = ["Technisches Problem", "Rechnungsfrage", "Account-Hilfe", "Allgemeines Feedback"]
MODELL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# --- FUNKTIONEN ---
def initialisiere_klassifikator():
    print(f"Lade das Klassifizierungsmodell '{MODELL_NAME}'...")
    print("Hinweis: Der erste Start kann einige Minuten dauern, da das Modell heruntergeladen wird.")
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
        print("    Interaktiver Support-Ticket-Klassifikator")
        print("="*60)
        print("Geben Sie eine Support-Anfrage ein, um sie zu klassifizieren.")
        print("Mögliche Kategorien sind:", TICKET_KATEGORIEN)
        print("Geben Sie 'exit' ein, um das Programm zu beenden.")
        print("-" * 60)

        # Diese Schleife läuft unendlich, bis der Benutzer 'exit' eingibt
        while True:
            # Auf eine Eingabe des Benutzers warten
            user_input = input("\nIhre Anfrage: ")

            # Prüfen, ob das Programm beendet werden soll
            if user_input.lower() == 'exit':
                print("Programm wird beendet. Auf Wiedersehen!")
                break
            
            # Klassifikation für die Benutzereingabe durchführen
            ergebnis = klassifiziere_text(ticket_classifier, user_input, TICKET_KATEGORIEN)
            
            if ergebnis:
                beste_kategorie = ergebnis['labels'][0]
                vertrauensscore = ergebnis['scores'][0]
                
                print(f"-> KI-Analyse: Diese Anfrage gehört am wahrscheinlichsten zur Kategorie '{beste_kategorie}' (Sicherheit: {vertrauensscore:.2%}).")
            else:
                print("-> Klassifikation fehlgeschlagen.")