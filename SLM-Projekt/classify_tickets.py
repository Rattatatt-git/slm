# -----------------------------------------------------------------------------
# Projekt: Automatische Klassifizierung von Support-Tickets
# Modell:  MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (Zero-Shot Classification)
# Zweck:   Minimal lauffähige Demonstration für die Projektaufgabe
# -----------------------------------------------------------------------------

# Benötigte Funktion 'pipeline' aus der transformers-Bibliothek importieren
from transformers import pipeline

# --- KONFIGURATION ---
# Die Kategorien, in die unsere Support-Tickets einsortiert werden sollen.
# Diese können beliebig angepasst oder erweitert werden.
TICKET_KATEGORIEN = ["Technisches Problem", "Rechnungsfrage", "Account-Hilfe", "Allgemeines Feedback"]

# Eine Liste von Beispiel-Tickets, die wir klassifizieren möchten.
BEISPIEL_TICKETS = [
    "Hallo, ich habe mein Passwort vergessen und kann mich nicht mehr einloggen.",
    "Der Betrag auf meiner letzten Rechnung vom 15. August scheint zu hoch zu sein.",
    "Seit dem letzten Update stürzt das Programm ständig ab, wenn ich eine Datei speichern will.",
    "Ich wollte nur mal sagen, dass mir das neue Design sehr gut gefällt. Weiter so!",
    "Wie kann ich meinen Benutzernamen ändern?"
]

# Name des zu verwendenden Modells von Hugging Face
MODELL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# --- FUNKTIONEN ---
def initialisiere_klassifikator():
    """
    Initialisiert die Zero-Shot-Klassifizierungs-Pipeline.
    Das Modell wird beim ersten Aufruf automatisch heruntergeladen.
    
    Returns:
        Eine Pipeline-Instanz, die für die Klassifizierung verwendet werden kann.
    """
    print(f"Lade das Klassifizierungsmodell '{MODELL_NAME}'...")
    print("Hinweis: Der erste Start kann einige Minuten dauern, da das Modell heruntergeladen wird.")
    try:
        # Die Pipeline wird mit der Aufgabe "zero-shot-classification" und unserem Modell initialisiert.
        classifier = pipeline("zero-shot-classification", model=MODELL_NAME)
        print("Modell erfolgreich geladen!")
        return classifier
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        print("Stellen Sie sicher, dass eine Internetverbindung besteht und die Bibliotheken korrekt installiert sind.")
        return None

def klassifiziere_text(classifier, text, kategorien):
    """
    Klassifiziert einen einzelnen Text in eine der gegebenen Kategorien.

    Args:
        classifier: Die initialisierte Klassifizierungs-Pipeline.
        text (str): Der zu klassifizierende Text.
        kategorien (list): Eine Liste der möglichen Kategorien.

    Returns:
        Ein Dictionary mit dem Ergebnis der Klassifikation.
    """
    if not classifier:
        return None
    
    # Führe die Klassifikation durch. 'multi_label=False' bedeutet, dass nur eine Kategorie
    # als wahrscheinlichste ausgewählt werden soll.
    return classifier(text, kategorien, multi_label=False)

# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    # 1. Pipeline initialisieren
    ticket_classifier = initialisiere_klassifikator()

    if ticket_classifier:
        print("\n" + "="*50)
        print("      Automatische Klassifizierung von Support-Tickets")
        print("="*50 + "\n")
        
        # 2. Jedes Beispiel-Ticket durchgehen und klassifizieren
        for i, ticket_text in enumerate(BEISPIEL_TICKETS):
            print(f"--- Ticket #{i+1} ---")
            print(f"Eingegangener Text: '{ticket_text}'")
            
            # Klassifikation durchführen
            ergebnis = klassifiziere_text(ticket_classifier, ticket_text, TICKET_KATEGORIEN)
            
            if ergebnis:
                # Das Ergebnis auswerten und ausgeben
                beste_kategorie = ergebnis['labels'][0]
                vertrauensscore = ergebnis['scores'][0]
                
                print(f"-> Ergebnis: Wahrscheinlichste Kategorie ist '{beste_kategorie}' (Sicherheit: {vertrauensscore:.2%}).")
            else:
                print("-> Klassifikation fehlgeschlagen.")
            
            print("-"*(20 + len(str(i+1))) + "\n")