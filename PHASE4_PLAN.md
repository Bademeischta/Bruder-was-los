# Phase 4: Der selbstlernende Zyklus

Dieses Dokument beschreibt die Architektur des selbstlernenden Zyklus, der die KI in die Lage versetzt, durch das Spielen gegen sich selbst kontinuierlich stärker zu werden. Der Zyklus besteht aus drei Kernmodulen, die von einer Hauptschleife orches­triert werden.

## 1. Das Selbstspiel-Modul (self_play)

Dieses Modul ist für die Generierung von qualitativ hochwertigen Trainingsdaten verantwortlich.

### Prozess

Eine einzelne Partie wird wie folgt generiert:

1.  **Initialisierung:** Eine neue Partie wird im `ChessCore` gestartet. Das aktuelle "beste" `ChessModel` wird geladen.
2.  **Iteratives Spielen:** Für jeden Zug in der Partie, bis das Spiel endet:
    a.  **MCTS-Suche:** Eine MCTS-Suche (`find_best_move`) wird für den aktuellen Brettzustand mit einer festgelegten Anzahl von Simulationen (z.B. 800) ausgeführt.
    b.  **Probabilistische Zugauswahl:** Um die Exploration zu fördern, wird der nächste Zug nicht immer deterministisch gewählt. Stattdessen werden die Besuchszahlen der Kindknoten des Wurzelknotens als Wahrscheinlichkeitsverteilung interpretiert. Für die ersten Züge einer Partie (z.B. die ersten 30) wird ein Zug proportional zu seiner Besuchszahl (`N(s,a)`) ausgewählt. Danach wird der Zug mit der höchsten Besuchszahl (deterministisch) gewählt, um die Partie auszunutzen.
    c.  **Datenspeicherung:** Der aktuelle Zustand, die berechnete MCTS-Policy und der spätere Partienausgang werden für das Training gespeichert.
    d.  **Zugausführung:** Der gewählte Zug wird im `ChessCore` ausgeführt.
3.  **Partienende:** Wenn die Partie beendet ist (`checkmate`, `stalemate` oder eine andere Regel), wird der finale Ausgang (+1 für Sieg, -1 für Niederlage, 0 für Unentschieden) bestimmt. Dieser Wert wird für alle in dieser Partie gespeicherten Trainingsdaten als `game_outcome` eingetragen.

### Datenspeicherung

Die Trainingsdaten werden als Liste von Tupeln gespeichert. Jedes Tupel repräsentiert einen einzelnen Zug in einer Partie und hat die folgende Struktur:

`(board_state_tensor, mcts_policy_vector, game_outcome)`

*   `board_state_tensor`: Der (21, 8, 8) Tensor, der den Brettzustand vor dem Zug repräsentiert (Output von `state_encoder.py`).
*   `mcts_policy_vector`: Ein 4672-dimensionaler Vektor. Die Wahrscheinlichkeit für jeden Zug ist proportional zu seiner Besuchszahl im MCTS (`N(s,a) / sum(N(s,b))`).
*   `game_outcome`: Ein Skalar (+1, -1, 0), der das Ergebnis der Partie aus der Perspektive des Spielers angibt, der am Zug war.

## 2. Das Trainings-Modul (train)

Dieses Modul verwendet die generierten Daten, um eine neue, verbesserte Version des neuronalen Netzes zu trainieren.

### Prozess

1.  **Datensammlung:** Sammle die Trainingsdaten aus einer großen Anzahl von Selbstspiel-Partien (z.B. die letzten 500.000 generierten Positionen).
2.  **Trainingsschleife:** Trainiere eine neue Instanz des `ChessModel` für eine feste Anzahl von Epochen mit den gesammelten Daten.
    *   Die Daten werden in zufällige Batches aufgeteilt.
    *   Ein Optimizer (z.B. Adam oder SGD mit Momentum) wird verwendet, um die Gewichte des Netzwerks zu aktualisieren.

### Loss-Funktion

Die kombinierte Loss-Funktion `L` ist entscheidend für das Training und setzt sich aus zwei Teilen zusammen:

`L = (v - z)^2 - π^T * log(p) + c * ||θ||^2`

*   **Value Loss:** `(v - z)^2` ist der Mean Squared Error zwischen dem vorhergesagten Wert `v` des Value Heads und dem tatsächlichen Partienausgang `z` (+1, -1, 0).
*   **Policy Loss:** `-π^T * log(p)` ist der Cross-Entropy-Loss zwischen der vom MCTS berechneten Policy `π` (der "Lehrer") und der vom Policy Head vorhergesagten Policy `p` (der "Schüler").
*   **Regularisierung:** `c * ||θ||^2` ist ein L2-Regularisierungsterm, um Overfitting zu verhindern, wobei `θ` die Gewichte des Netzwerks sind.

## 3. Das Evaluations-Modul (evaluate)

Dieses Modul stellt sicher, dass nur objektiv bessere Modelle zum neuen Standard werden.

### Prozess

1.  **Turnier:** Das neu trainierte Modell tritt gegen das aktuell beste Modell in einem Turnier von z.B. 100 Partien an.
2.  **Bedingungen:**
    *   Beide Modelle verwenden MCTS mit der gleichen Anzahl von Simulationen.
    *   Die Zugauswahl erfolgt immer deterministisch (der Zug mit den meisten Besuchen wird gewählt), um die reine Spielstärke zu messen.
    *   Die Farben werden abwechselnd zugewiesen.
3.  **Ergebnis:** Die Gewinnrate des neuen Modells wird berechnet.

### Akzeptanzkriterium

Das neue Modell ersetzt das alte als "bestes" Modell nur dann, wenn es eine signifikant höhere Gewinnrate erzielt, z.B. **> 55%**. Dieser Schwellenwert verhindert, dass Modelle aufgrund von Zufallsschwankungen ersetzt werden.

## 4. Die Hauptschleife (main_loop)

Diese Schleife orchestriert den gesamten Prozess und treibt den selbstlernenden Zyklus an.

```
WHILE True:
  # 1. Generiere eine neue Generation von Selbstspiel-Daten
  #    mit dem aktuell besten Modell.
  self_play_data = self_play.generate_games(best_model, num_games=1000)

  # 2. Trainiere ein neues Modell mit den gesammelten Daten.
  new_model = train.train_model(self_play_data)

  # 3. Evaluiere das neue Modell gegen das beste Modell.
  win_rate = evaluate.run_tournament(new_model, best_model, num_matches=100)

  # 4. Wenn das neue Modell besser ist, ersetze das alte.
  IF win_rate > 0.55:
    best_model.save("models/archive/model_{timestamp}.pth")
    new_model.save("models/best_model.pth")
    best_model = new_model
```