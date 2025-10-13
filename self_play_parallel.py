import torch
import torch.multiprocessing as mp
from transformer_network import ChessTransformer
from self_play import play_game
from functools import partial
import os

def run_parallel_games(model: ChessTransformer, num_games: int, num_simulations: int, num_processes: int = None) -> list:
    """
    Führt mehrere Selbstspiel-Partien parallel aus, um die Datengenerierung
    zu beschleunigen.

    Args:
        model (ChessModel): Das neuronale Netz, das die MCTS-Suche leitet.
        num_games (int): Die Gesamtzahl der zu spielenden Partien.
        num_simulations (int): Die Anzahl der MCTS-Simulationen pro Zug.
        num_processes (int, optional): Die Anzahl der zu verwendenden CPU-Kerne.
                                       Defaults to os.cpu_count().

    Returns:
        list: Eine aggregierte Liste von Trainings-Tupeln aus allen Partien.
    """
    if num_processes is None:
        num_processes = os.cpu_count()

    # Stellen Sie sicher, dass das Modell im Speicher geteilt werden kann
    # zwischen den Prozessen, ohne dass jeder eine eigene Kopie lädt.
    model.share_memory()

    # Erstelle eine Liste von Argumenten-Tupeln für jede Partie
    args_list = [(model, num_simulations, 30, i) for i in range(num_games)]

    print(f"Starte {num_games} Partien auf {num_processes} Kernen...")

    with mp.Pool(processes=num_processes) as pool:
        # Führe die Funktion parallel mit starmap aus
        results = pool.starmap(play_game, args_list)

    # Sammle die Ergebnisse aus allen Prozessen
    all_training_data = []
    for game_data in results:
        all_training_data.extend(game_data)

    return all_training_data


if __name__ == '__main__':
    # Beispiel für die parallele Generierung von Trainingsdaten
    print("Starte Beispiel für paralleles Selbstspiel...")

    # Wichtig: Setze den Start-Method für Multiprocessing. 'fork' kann bei CUDA zu Problemen führen.
    # 'spawn' ist sicherer und plattformunabhängiger.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    model = ChessTransformer()

    # Führe 4 Partien parallel aus
    num_parallel_games = 4
    training_data = run_parallel_games(model, num_games=num_parallel_games, num_simulations=50)

    print(f"\nParalleles Selbstspiel beendet. {len(training_data)} Trainingsdatensätze generiert.")

    if training_data:
        print(f"Durchschnittliche Anzahl an Datensätzen pro Partie: {len(training_data) / num_parallel_games:.1f}")
        assert len(training_data) > 0
        print("Sanity-Check erfolgreich!")