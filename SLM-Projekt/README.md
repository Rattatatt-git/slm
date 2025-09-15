# Projekt: Automatische Klassifizierung von Support-Tickets

Dieses Projekt ist eine minimale Demonstration, wie ein kleines KI-Sprachmodell genutzt werden kann, um Support-Anfragen automatisch zu kategorisieren. Die Ausführung erfolgt in Visual Studio Code.

## Kurzanleitung zum Starten

### 1. Projekt in VS Code öffnen
- Starten Sie Visual Studio Code.
- Gehen Sie auf `Datei > Ordner öffnen...` und wählen Sie den Ordner `SLM-Projekt` aus.

### 2. Terminal in VS Code öffnen
- Öffnen Sie das integrierte Terminal in VS Code über das Menü `Terminal > Neues Terminal` oder mit dem Shortcut `Strg+Ö`.

### 3. Python-Umgebung einrichten und Programm starten
- Es wird empfohlen, eine virtuelle Umgebung zu verwenden, um die Projekt-Bibliotheken sauber zu isolieren. Führen Sie die folgenden Befehle nacheinander im Terminal aus:

- **Umgebung erstellen:**
  ```bash
  py -m venv umgebung
  ```

- **Umgebung aktivieren (für Windows):**
  ```bash
  .\umgebung\scripts\activate
  ```
  *(Für macOS / Linux wäre der Befehl `source umgebung/bin/activate`)*

- **Abhängigkeiten installieren:**
  ```bash
  pip install -r requirements.txt
  ```

- **Anwendung starten:**
  ```bash
  py classify_tickets.py
  ```

### 4. Interaktion mit dem Programm

Nachdem Sie den Befehl `py classify_tickets.py` ausgeführt haben, passiert Folgendes:

1.  **Modelle werden geladen:** Das Programm beginnt damit, die benötigten KI-Modelle herunterzuladen (nur beim allerersten Start) und zu laden. Dies kann je nach Internetverbindung einige Minuten dauern. Sie sehen im Terminal entsprechende Lade-Meldungen.

2.  **Programm ist startklar:** Sobald alles geladen ist, erscheint eine Startmeldung und das Programm wartet auf Ihre Eingabe. Sie erkennen das an der Zeile:
    ```
    Ihre Anfrage: 
    ```

3.  **Anfragen testen:** Geben Sie jetzt einfach eine beliebige Support-Anfrage ein (z.B., "Wie hoch ist meine Rechnung?" oder "Welchen Betrag muss ich überweisen?") und drücken Sie Enter.

4.  **Ergebnis ansehen:** Die KI analysiert Ihren Text und gibt das Ergebnis direkt im Terminal aus. Sie sehen, welche Kategorie mit welcher Sicherheit erkannt wurde.

5.  **Beenden des Programms:** Um das Programm zu beenden, geben Sie einfach das Wort `exit` ein und drücken Sie Enter.

Viel Spaß beim Testen!