import torch
from neural_network import ChessModel
from self_play_parallel import run_parallel_games
from train import train_model
from evaluate import run_tournament
import os
import time

def main_loop(
    num_iterations: int,
    num_games_per_iteration: int,
    num_simulations_per_move: int,
    num_eval_games: int,
    epochs_per_training: int,
    acceptance_threshold: float = 0.55
):
    """
    Orchestriert den vollständigen, selbstlernenden Zyklus.

    Args:
        num_iterations (int): Die Anzahl der Zyklen (Selbstspiel -> Training -> Evaluation), die durchlaufen werden sollen.
        num_games_per_iteration (int): Die Anzahl der Selbstspiel-Partien pro Iteration.
        num_simulations_per_move (int): Die Anzahl der MCTS-Simulationen pro Zug.
        num_eval_games (int): Die Anzahl der Partien im Evaluations-Turnier.
        epochs_per_training (int): Die Anzahl der Epochen pro Trainingsphase.
        acceptance_threshold (float): Die Mindestgewinnrate, um ein neues Modell zu akzeptieren.
    """

    # Stelle sicher, dass die Verzeichnisse für die Modelle existieren
    os.makedirs("models/archive", exist_ok=True)

    # Initialisiere oder lade das beste Modell
    best_model_path = "models/best_model.pth"
    if os.path.exists(best_model_path):
        print("Lade bestes Modell...")
        best_model = ChessModel()
        best_model.load_state_dict(torch.load(best_model_path))
    else:
        print("Erstelle initiales Modell...")
        best_model = ChessModel()
        torch.save(best_model.state_dict(), best_model_path)

    for i in range(num_iterations):
        print(f"\n--- Starte Iteration {i+1}/{num_iterations} ---")

        # 1. Selbstspiel-Phase
        print(f"Generiere {num_games_per_iteration} neue Partien parallel...")
        training_data = run_parallel_games(best_model, num_games=num_games_per_iteration, num_simulations=num_simulations_per_move)

        # 2. Trainings-Phase
        print(f"\nTrainiere neues Modell mit {len(training_data)} Positionen...")
        new_model = ChessModel()
        new_model.load_state_dict(best_model.state_dict()) # Starte mit den Gewichten des besten Modells
        train_model(new_model, training_data, epochs=epochs_per_training)

        # 3. Evaluations-Phase
        print(f"\nEvaluiere neues Modell gegen bestes Modell in {num_eval_games} Partien...")
        win_rate = run_tournament(new_model, best_model, num_matches=num_eval_games, num_simulations=num_simulations_per_move)

        print(f"\nErgebnis der Evaluation: Gewinnrate des neuen Modells: {win_rate*100:.1f}%")

        # 4. Akzeptanz-Phase
        if win_rate > acceptance_threshold:
            timestamp = int(time.time())
            archive_path = f"models/archive/model_{timestamp}.pth"
            print(f"Neues Modell ist besser! Archiviere altes Modell nach '{archive_path}' und speichere neues als 'best_model.pth'.")
            torch.save(best_model.state_dict(), archive_path)
            torch.save(new_model.state_dict(), best_model_path)
            best_model = new_model
        else:
            print("Neues Modell hat das Akzeptanzkriterium nicht erfüllt. Verwerfe neues Modell.")

    print("\nAlle Iterationen abgeschlossen.")


if __name__ == '__main__':
    # Führe eine einzelne, verkürzte Iteration zu Demonstrationszwecken aus
    print("Starte eine verkürzte Demo der Hauptschleife...")
    main_loop(
        num_iterations=1,
        num_games_per_iteration=2,       # Nur 2 Partien
        num_simulations_per_move=10,     # Nur 10 Simulationen
        num_eval_games=2,                # Nur 2 Evaluationspartien
        epochs_per_training=3            # Nur 3 Epochen
    )