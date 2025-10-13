# AlphaZero-Style Chess AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bademeischta/Bruder-was-los)

This project is a complete, from-scratch implementation of a chess engine based on the principles of DeepMind's AlphaZero. The AI learns to play chess solely by playing against itself, using a sophisticated combination of a deep neural network and a Monte-Carlo Tree Search (MCTS) algorithm.

## Architektur

The project is divided into several modular components that work together to create the self-learning cycle.

### Core Components

*   **`chess_core.py`**: A robust wrapper around the `python-chess` library. It provides a clean and controlled interface for game state manipulation, move validation, and status checking.
*   **`neural_network.py`**: The "brain" of the AI. It's a PyTorch implementation of a deep residual neural network (`ChessModel`) with a dual-head architecture:
    *   **Policy Head**: Predicts the probability of the best move from a given position.
    *   **Value Head**: Estimates the probability of winning from a given position.
*   **`state_encoder.py` & `move_translator.py`**: These are crucial helper modules that translate the game state into a format the neural network can understand (a 21-channel tensor) and translate the network's policy output back into legal chess moves.
*   **`mcts.py`**: The "thinker" of the AI. It implements the Monte-Carlo Tree Search algorithm. Instead of brute-forcing moves, MCTS intelligently explores the most promising move sequences, guided by the predictions of the neural network.

### The Self-Learning Cycle

The AI's ability to improve comes from a continuous loop of three processes, orchestrated by `main_loop.py`:

1.  **Parallel Self-Play (`self_play_parallel.py`)**: To drastically speed up data generation, the system plays hundreds of games against itself in parallel, utilizing all available CPU cores. For each move, an MCTS search is performed to find the best action. The history of these games is stored as training data.
2.  **Training (`train.py`)**: A new neural network is trained using the data generated during self-play. The network learns to predict two things:
    *   The move probabilities calculated by the MCTS (the "teacher").
    *   The actual game outcome (win/loss/draw).
3.  **Evaluation (`evaluate.py`)**: The newly trained network plays a tournament against the previous best network. If it wins by a significant margin (e.g., >55% win rate), it becomes the new "best" network.

This cycle repeats, allowing the AI to incrementally improve its understanding of chess and its playing strength.

## Installation

To run this project, you need Python 3.9+ and a GPU for any serious training.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Bademeischta/Bruder-was-los.git
    cd Bruder-was-los
    ```

2.  **Install the dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Verwendung

### Running Tests

To ensure all components are working correctly, you can run the comprehensive test suite:
```bash
python -m unittest discover
```

### Running Individual Modules

Each core module contains a simple demonstration script in its `if __name__ == '__main__':` block. You can run them individually to see them in action.

```bash
# Demonstrate the MCTS search finding a move
python main.py

# Demonstrate the self-play data generation for one game
python self_play.py

# Demonstrate the training loop with dummy data
python train.py

# Demonstrate the evaluation tournament between two models
python evaluate.py
```

### Starting the Full Training Loop

The main entry point to start the autonomous, self-learning cycle is `main_loop.py`.

**Warning**: This is a computationally intensive process that will run for a very long time and requires a powerful GPU. The parameters in the script are set for a short demonstration. For a real training run, these values (especially `num_games_per_iteration` and `num_simulations_per_move`) should be significantly increased.

```bash
python main_loop.py
```

## Running in Google Colab

You can run this project in a Google Colab notebook, which provides free access to GPUs.

1.  Click the "Open in Colab" badge at the top of this README.
2.  Once in the Colab environment, you can execute shell commands by prefixing them with `!`.
3.  First, clone the repository and install the dependencies in your Colab instance:
    ```python
    !git clone https://github.com/Bademeischta/Bruder-was-los.git
    %cd Bruder-was-los
    !pip install -r requirements.txt
    ```
4.  Now you can run the tests or start the main training loop as described above:
    ```python
    # Run all tests to verify the environment
    !python -m unittest discover

    # Start the main loop (make sure to use a GPU runtime in Colab)
    !python main_loop.py
    ```

## Funktionsweise und erweiterte Nutzung

### Funktioniert das Training wirklich?

**Ja, der Code ist eine vollständige und funktionierende Implementierung des AlphaZero-Algorithmus.** Der selbstlernende Zyklus in `main_loop.py` ist voll funktionsfähig und orchestriert die Phasen korrekt:

1.  **Selbstspiel:** Das System spielt Partien gegen sich selbst und generiert Trainingsdaten.
2.  **Training:** Das `train.py`-Modul **trainiert das neuronale Netz auch wirklich.** Die Ausgabe mit dem sinkenden Loss (z.B. `Epoch 1/5, Loss: 7.8028` -> `Epoch 5/5, Loss: 5.2807`) ist der Beweis dafür. In jeder Epoche passt der Optimizer die Gewichte des `ChessModel` an, um die Differenz zwischen seinen Vorhersagen und den "besseren" Daten aus der MCTS-Suche zu minimieren. Das Modell lernt also tatsächlich aus seinen Erfahrungen.
3.  **Evaluation:** Das System vergleicht das neu trainierte Modell mit dem bisher besten und ersetzt es bei signifikanter Verbesserung.

Der entscheidende Punkt ist der **Maßstab**: Die Demonstrationsläufe werden mit extrem kleinen Parametern ausgeführt (wenige Simulationen, wenige Partien). Das beweist die *Funktionsfähigkeit der Architektur*, aber reicht nicht aus, um eine starke Schach-KI zu erschaffen. Sie haben eine voll funktionsfähige Maschine gebaut; jetzt braucht sie nur noch die Zeit und die Rechenleistung (den "Treibstoff"), um wirklich intelligent zu werden.

### Hyperparameter für ein echtes Training

Um ein ernsthaftes Training durchzuführen, müssen die Parameter in `main_loop.py` deutlich erhöht werden:

*   `num_iterations`: Hunderte oder Tausende, um eine kontinuierliche Verbesserung zu ermöglichen.
*   `num_games_per_iteration`: Eine große Zahl (z.B. 25.000, wie im AlphaZero-Paper), um eine vielfältige Datenbasis für jede Trainingsphase zu schaffen.
*   `num_simulations_per_move`: Der wichtigste Parameter für die Spielstärke. Werte von 800, 1600 oder mehr sind hier üblich.
*   `epochs_per_training`: Genügend Epochen, damit das Netz aus den neuen Daten lernen kann, ohne zu overfitten.

### Analyse einer einzelnen Stellung

Sie können ein trainiertes Modell verwenden, um die beste Aktion für eine beliebige Schachstellung zu finden. Erstellen Sie dafür ein Skript `analyze.py`:

```python
# analyze.py
import chess
import torch
from neural_network import ChessModel
from mcts import MCTS

# 1. Laden Sie Ihr bestes trainiertes Modell
model = ChessModel()
model.load_state_dict(torch.load("models/best_model.pth"))

# 2. Definieren Sie die Stellung mittels FEN-String
fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" # Beispiel: Sizilianische Verteidigung
board = chess.Board(fen)

# 3. Führen Sie die MCTS-Suche aus
mcts = MCTS(model)
num_simulations = 800  # Eine hohe Anzahl für eine gute Analyse
best_move = mcts.find_best_move(board, num_simulations)

print(f"Analyse für FEN: {fen}")
print(f"Bester gefundener Zug nach {num_simulations} Simulationen: {best_move.uci()}")
```

### Visualisierung einer Partie

Um eine Selbstspiel-Partie live zu verfolgen, können Sie die `play_game`-Funktion in `self_play.py` leicht anpassen. Fügen Sie einfach eine `print(board)`-Anweisung innerhalb der `while`-Schleife hinzu:

```python
# In self_play.py, innerhalb der play_game-Funktion:
# ...
    while not board.is_game_over():
        # ... (MCTS-Suche)

        # ... (Zugauswahl)
        board.push(move)

        # Fügen Sie diese Zeilen hinzu, um das Brett zu visualisieren
        print("\n" + str(board))
        print(f"Zug: {move.uci()}, Zugnummer: {board.fullmove_number}, Spieler am Zug: {'Weiß' if board.turn else 'Schwarz'}")
```

---

## Interaktive Nutzung

### Schnelles Training auf Colab

Der `main_loop.py` ist für ein langes, tiefgehendes Training konzipiert. Um in kurzer Zeit (ca. 10-15 Minuten auf einer Colab-GPU) ein spielbares Modell zu erhalten, verwenden Sie das optimierte Skript `train_fast.py`.

```bash
# Führt einen optimierten Trainingszyklus aus
python train_fast.py
```
Dieser Prozess erzeugt eine `models/best_model.pth`-Datei, die Sie für das Spiel gegen die KI verwenden können.

### Gegen die KI spielen

Sobald ein trainiertes Modell (`best_model.pth`) vorhanden ist, können Sie mit dem Skript `play_vs_ai.py` eine interaktive Partie auf der Konsole spielen.

```bash
python play_vs_ai.py
```

Das Skript wird Sie auffordern, Ihre Farbe zu wählen (Weiß oder Schwarz) und Ihre Züge im UCI-Format (z.B. `e2e4`) einzugeben.