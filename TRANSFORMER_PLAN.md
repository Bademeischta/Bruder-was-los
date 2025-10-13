# Phase 5: Radikale Neuausrichtung - Die Transformer-Architektur

## 1. Motivation: Jenseits räumlicher Muster

Die bisherige CNN-Architektur war erfolgreich darin, lokale, räumliche Muster auf dem Schachbrett zu erkennen. Wie in der Analyse korrekt angemerkt, ist Schach jedoch fundamental ein Spiel der **Beziehungen** zwischen Figuren, nicht nur der räumlichen Nähe. Ein fianchetto-Läufer und eine schwache Königsstellung beeinflussen sich über das gesamte Brett hinweg.

Um von einer rein statistischen zu einer quasi-konzeptionellen Analyse überzugehen, ersetzen wir die CNN-Architektur durch einen **Transformer**. Der Attention-Mechanismus des Transformers ist explizit dafür konzipiert, die Wichtigkeit und Beziehung jedes Elements in einer Sequenz zu allen anderen Elementen zu lernen. Dies ermöglicht es dem Modell, ein tieferes, kontextuelles Verständnis der Brettstellung zu entwickeln.

## 2. Tokenisierung: Das Brett als Sequenz

Wir geben die 8x8-Bildrepräsentation auf. Stattdessen modellieren wir das Brett als eine Sequenz von Tokens. Jeder Token repräsentiert eine Informationseinheit.

**Input-Sequenz:** Die maximale Länge der Sequenz beträgt 38 Tokens.
`[CLS, SideToMove, CastlingRights(4), Piece1, Piece2, ..., Piece32]`

*   **`[CLS]`-Token (1 Token):** Ein spezieller Token am Anfang der Sequenz. Die finale Ausgabe des Transformers an dieser Position wird für die `Value`-Bewertung der gesamten Stellung verwendet.
*   **`SideToMove`-Token (1 Token):** Ein lernbares Embedding, das angibt, ob Weiß oder Schwarz am Zug ist.
*   **`CastlingRights`-Token (4 Tokens):** Vier separate, lernbare Embeddings, die die vier Rochaderechte (Weiß Königseite, Weiß Damenseite, etc.) repräsentieren.
*   **Piece-Tokens (max. 32 Tokens):** Jeder Token repräsentiert eine einzelne Figur auf dem Brett.

**Embedding eines Piece-Tokens:** Jede Figur wird in einen Vektor (`d_model`) eingebettet, der sich aus drei Teilen zusammensetzt:
1.  **Piece Embedding:** Ein lernbares Embedding für den Figurentyp und die Farbe (z.B. "weißer Bauer", "schwarze Dame"). (12 mögliche Werte)
2.  **Position Embedding:** Ein lernbares Embedding für jedes der 64 Felder. Dies gibt dem Modell die Information, *wo* sich die Figur befindet.
3.  Die beiden Embeddings werden addiert, um die finale Repräsentation des Piece-Tokens zu erzeugen.

Dieser Ansatz ermöglicht es dem Modell, die Identität und Position jeder Figur als eine einzige, reiche Repräsentation zu verarbeiten.

## 3. Netzwerkarchitektur: Der Transformer-Encoder

Das Herzstück des neuen Modells ist ein Transformer-Encoder.

*   **Embedding Layer:** Wandelt die oben beschriebene Token-Sequenz in eine Sequenz von Vektoren der Dimension `d_model` (z.B. 256) um.
*   **Transformer-Encoder-Schichten (z.B. 6 Schichten):** Ein Stapel von Standard-Transformer-Encodern. Jede Schicht besteht aus:
    *   **Multi-Head Self-Attention:** Hier lernt das Modell die Beziehungen. Jeder Token (jede Figur) "achtet" auf jeden anderen Token und gewichtet, wie wichtig die Beziehung für das Verständnis der Stellung ist.
    *   **Feed-Forward Network:** Eine kleine MLP, die auf jeden Token in der Sequenz angewendet wird.
*   Die Verbindungen zwischen den Sub-Layern verwenden Residual Connections und Layer Normalization.

## 4. Output-Heads: Policy und Value

Die Ausgabe des Transformer-Encoders (eine Sequenz von Vektoren) wird an zwei spezialisierte Köpfe weitergeleitet.

*   **Value Head:**
    *   Nimmt die finale Ausgabe des `[CLS]`-Tokens.
    *   Leitet sie durch eine kleine MLP (z.B. `Linear -> ReLU -> Linear -> Tanh`).
    *   Gibt einen einzelnen skalaren Wert zwischen -1 und 1 aus, der die Gewinnwahrscheinlichkeit repräsentiert.

*   **Policy Head:**
    *   Dieser Kopf ist entscheidend für die Verbindung zur MCTS.
    *   Er nimmt die finalen Ausgabe-Vektoren für alle **Piece-Tokens**.
    *   Für jeden dieser Figuren-Vektoren sagt eine lineare Schicht die Logits für die **73 möglichen Aktionstypen** (Queen-Züge, Springer-Züge, Unterverwandlungen) voraus, die von dieser Figur ausgehen könnten.
    *   Das Ergebnis ist eine Ausgabe der Dimension `(Anzahl_Figuren, 73)`.
    *   Diese "lokalen" Policies werden dann mithilfe des `move_translator`-Moduls zu dem finalen 4672-dimensionalen Policy-Vektor zusammengesetzt, den die MCTS erwartet.

Dieser Ansatz ermöglicht es dem Modell, für jede Figur zu entscheiden, was ihre besten Aktionen sind, basierend auf dem globalen Kontext, den der Attention-Mechanismus geschaffen hat.

## 5. Zusammenfassung der Änderungen

*   **Erstellen:** `TRANSFORMER_PLAN.md`, `board_tokenizer.py`, `transformer_network.py`, `test_board_tokenizer.py`, `test_transformer_network.py`.
*   **Ersetzen/Anpassen:**
    *   `state_encoder.py` wird durch `board_tokenizer.py` ersetzt.
    *   `neural_network.py` wird durch `transformer_network.py` ersetzt.
    *   `mcts.py`, `train.py` und die anderen Hauptmodule werden angepasst, um die neue, sequenzbasierte Ein- und Ausgabe zu verarbeiten.
    *   Bestehende Tests werden angepasst oder entfernt.