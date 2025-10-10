import chess
import numpy as np
import torch
from chess_core import ChessCore
from neural_network import ChessModel
from mcts import MCTS, Node
from state_encoder import board_to_tensor
from move_translator import move_to_index

def play_game(model: ChessModel, num_simulations: int, exploration_moves: int = 30) -> list:
    """
    Spielt eine einzelne Partie von Anfang bis Ende gegen sich selbst und
    sammelt dabei Trainingsdaten.

    Args:
        model (ChessModel): Das neuronale Netz, das die MCTS-Suche leitet.
        num_simulations (int): Die Anzahl der MCTS-Simulationen pro Zug.
        exploration_moves (int): Die Anzahl der Züge am Anfang der Partie,
                                 bei denen der Zug probabilistisch ausgewählt wird.

    Returns:
        list: Eine Liste von Trainings-Tupeln:
              [(state_tensor, policy_vector, outcome), ...].
              Der 'outcome' wird erst am Ende der Partie eingetragen.
    """
    board = chess.Board()
    mcts = MCTS(model)

    game_data = []

    while not board.is_game_over():
        # Führe MCTS aus, um den besten Zug und die Policy zu finden
        root = Node(state=board.copy())
        for _ in range(num_simulations):
            mcts._run_simulation(root)

        # Erstelle den Policy-Vektor basierend auf den Besuchszahlen
        policy_vector = np.zeros(4672, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits > 0:
            for move, child in root.children.items():
                index = move_to_index(move, board)
                policy_vector[index] = child.visit_count / total_visits

        # Speichere den Zustand und die berechnete Policy
        state_tensor = board_to_tensor(board)
        game_data.append([state_tensor, policy_vector, 0.0]) # Outcome wird später gesetzt

        # Wähle den nächsten Zug aus
        if board.fullmove_number < exploration_moves:
            # Probabilistische Auswahl basierend auf Besuchszahlen
            moves = list(root.children.keys())
            visit_counts = np.array([root.children[m].visit_count for m in moves], dtype=np.float32)
            probabilities = visit_counts / visit_counts.sum()
            move = np.random.choice(moves, p=probabilities)
        else:
            # Deterministische Auswahl (meistbesuchter Zug)
            move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)

        board.push(move)

    # Bestimme den Ausgang der Partie
    outcome = board.outcome()
    if outcome is None: # Sollte nicht passieren, aber zur Sicherheit
        final_outcome = 0.0
    elif outcome.winner == chess.WHITE:
        final_outcome = 1.0
    elif outcome.winner == chess.BLACK:
        final_outcome = -1.0
    else: # Unentschieden
        final_outcome = 0.0

    # Trage den finalen Ausgang in alle Trainingsdaten dieser Partie ein
    # Der Wert muss aus der Perspektive des Spielers am Zug sein
    current_player = chess.WHITE
    for i in range(len(game_data)):
        if current_player == chess.WHITE:
            game_data[i][2] = final_outcome
        else:
            game_data[i][2] = -final_outcome
        # Wechsel die Perspektive für den nächsten gespeicherten Zustand
        current_player = not current_player

    return game_data

if __name__ == '__main__':
    # Beispiel für die Generierung von Trainingsdaten aus einer Partie
    print("Starte Beispiel für Selbstspiel...")
    model = ChessModel()

    # Führe eine kurze Partie mit wenigen Simulationen aus
    training_data = play_game(model, num_simulations=10, exploration_moves=10)

    print(f"Partie beendet. {len(training_data)} Trainingsdatensätze generiert.")

    # Überprüfe den ersten und letzten Datensatz
    if training_data:
        first_state, first_policy, first_outcome = training_data[0]
        last_state, last_policy, last_outcome = training_data[-1]

        print("\n--- Erster Datensatz ---")
        print(f"State Tensor Shape: {first_state.shape}")
        print(f"Policy Vector Shape: {first_policy.shape}")
        print(f"Summe der Policy-Wahrscheinlichkeiten: {np.sum(first_policy):.2f}")
        print(f"Partienausgang (aus Perspektive von Weiß): {first_outcome}")

        print("\n--- Letzter Datensatz ---")
        print(f"Partienausgang (aus Perspektive des letzten Spielers): {last_outcome}")

        assert first_state.shape == (21, 8, 8)
        assert first_policy.shape == (4672,)
        assert np.isclose(np.sum(first_policy), 1.0)
        print("\nSanity-Checks erfolgreich!")