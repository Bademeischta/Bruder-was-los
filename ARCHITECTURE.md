# Architektur-Dokumentation

Dieses Dokument hält die zentralen Design- und Architekturentscheidungen für das Schach-KI-Projekt fest.

## Phase 2: Neuronales Netzwerk

### Framework-Wahl

**Gewähltes Framework:** PyTorch

**Begründung:**

1.  **Flexibilität und Forschungsnähe:** PyTorch ist in der akademischen Forschung, insbesondere im Bereich Reinforcement Learning und bei komplexen Architekturen wie AlphaZero, das führende Framework. Seine "define-by-run"-Philosophie ermöglicht eine intuitive und flexible Implementierung von komplexen Modellen, was für dieses Projekt entscheidend ist.
2.  **Debugging:** Der dynamische Berechnungsgraph von PyTorch erleichtert das Debugging erheblich. Man kann Standard-Python-Tools wie `pdb` verwenden, um den Code schrittweise zu durchlaufen und den Zustand von Tensoren zu inspizieren, was die Entwicklung beschleunigt und die Fehleranfälligkeit reduziert.
3.  **Community und Ressourcen:** Aufgrund seiner Dominanz in der Forschung gibt es eine Fülle von Open-Source-Implementierungen, Tutorials und wissenschaftlichen Arbeiten, die PyTorch verwenden. Dies ist ein unschätzbarer Vorteil, wenn man sich an State-of-the-Art-Architekturen orientiert.

Obwohl TensorFlow exzellente Werkzeuge für die Skalierung und das Deployment bietet, überwiegen für die hier geforderte Implementierung einer komplexen, forschungsnahen Netzwerkarchitektur die Vorteile von PyTorch in Bezug auf Flexibilität und Entwicklerfreundlichkeit.

### Datenrepräsentation (Input-Tensor)

Um den Zustand des Spiels für das Neuronale Netz verarbeitbar zu machen, wird der Zustand des `chess.Board`-Objekts in einen numerischen Tensor umgewandelt. Die Architektur orientiert sich am AlphaZero-Ansatz und verwendet eine Stapelung von 8x8-Ebenen (Planes).

**Tensor-Dimension:** `(21, 8, 8)`

**Aufschlüsselung der 21 Kanäle:**

1.  **Figurenpositionen (12 Kanäle):**
    *   **Ebenen 0-5:** Figuren des aktuellen Spielers (Bauer, Springer, Läufer, Turm, Dame, König). Eine `1` auf einem Feld markiert die Präsenz der Figur, sonst `0`.
    *   **Ebenen 6-11:** Figuren des Gegners (Bauer, Springer, Läufer, Turm, Dame, König).

2.  **Spielzustand / Meta-Informationen (7 Kanäle):**
    *   **Ebene 12:** Rochaderecht Weiß, Königseite (komplett `1`, falls Recht besteht, sonst `0`).
    *   **Ebene 13:** Rochaderecht Weiß, Damenseite (komplett `1`, falls Recht besteht, sonst `0`).
    *   **Ebene 14:** Rochaderecht Schwarz, Königseite (komplett `1`, falls Recht besteht, sonst `0`).
    *   **Ebene 15:** Rochaderecht Schwarz, Damenseite (komplett `1`, falls Recht besteht, sonst `0`).
    *   **Ebene 16:** Spieler am Zug (komplett `1` für Weiß, `0` für Schwarz).
    *   **Ebene 17:** Gesamtzahl der Züge (normalisierter Wert, z.B. `board.fullmove_number / 200`).
    *   **Ebene 18:** 50-Züge-Regel-Zähler (normalisierter Wert, `board.halfmove_clock / 100`).

3.  **Zugwiederholung (2 Kanäle):**
    *   **Ebene 19:** Eine Ebene, die `1` ist, wenn die aktuelle Brettstellung genau einmal zuvor aufgetreten ist.
    *   **Ebene 20:** Eine Ebene, die `1` ist, wenn die aktuelle Brettstellung genau zweimal oder öfter zuvor aufgetreten ist.

### Netzwerkarchitektur (AlphaZero-Stil)

Das Netzwerk besteht aus einem gemeinsamen "Body" und zwei spezialisierten "Heads" für Policy und Value.

1.  **Gemeinsamer Body (Feature Extractor):**
    *   **Eingangs-Block:**
        *   `Conv2d(in_channels=21, out_channels=256, kernel_size=3, padding=1)`
        *   `BatchNorm2d(256)`
        *   `ReLU`
    *   **Residuelle Blöcke:**
        *   Ein Stapel von **19 residuellen Blöcken**.
        *   Jeder Block folgt dem Schema: `Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add(Input) -> ReLU`. Alle Convolutional Layer im Body haben 256 Kanäle.

2.  **Policy Head (Vorhersage der Zugwahrscheinlichkeiten):**
    *   Nimmt den `(256, 8, 8)`-Tensor vom Body entgegen.
    *   **Conv-Block:**
        *   `Conv2d(in_channels=256, out_channels=2, kernel_size=1)`
        *   `BatchNorm2d(2)`
        *   `ReLU`
    *   **Fully-Connected Layer:**
        *   Abflachen des Tensors auf die Größe `2 * 8 * 8 = 128`.
        *   `Linear(128, 4672)`
    *   **Output:** Ein Logit-Vektor der Größe 4672, der die Wahrscheinlichkeitsverteilung über alle möglichen Züge repräsentiert.

3.  **Value Head (Vorhersage der Positionsbewertung):**
    *   Nimmt den `(256, 8, 8)`-Tensor vom Body entgegen.
    *   **Conv-Block:**
        *   `Conv2d(in_channels=256, out_channels=1, kernel_size=1)`
        *   `BatchNorm2d(1)`
        *   `ReLU`
    *   **Fully-Connected Layers:**
        *   Abflachen des Tensors auf die Größe `1 * 8 * 8 = 64`.
        *   `Linear(64, 256)`
        *   `ReLU`
        *   `Linear(256, 1)`
    *   **Output:** Ein einzelner skalarer Wert, der durch eine `tanh`-Aktivierungsfunktion auf den Bereich `[-1, 1]` normalisiert wird.