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

1.  **Self-Play (`self_play.py`)**: The current best neural network plays thousands of games against itself. For each move, an MCTS search is performed to find the best action. The history of these games (states, MCTS-derived policies, and final game outcomes) is stored as training data.
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