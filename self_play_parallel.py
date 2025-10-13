import torch
import torch.multiprocessing as mp
from neural_network import ChessModel
from self_play import play_game
from functools import partial
import os

def run_parallel_games(model: ChessModel, num_games: int, num_simulations: int, num_processes: int = None) -> list:
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

    # `partial` wird verwendet, um die `play_game`-Funktion mit festen
    # Argumenten (model, num_simulations) zu versehen.
    # Der Pool wird dann nur noch die variablen Teile (die Spielnummer) übergeben.
    game_function = partial(play_game, model, num_simulations)

    print(f"Starte {num_games} Partien auf {num_processes} Kernen...")

    with mp.Pool(processes=num_processes) as pool:
        # Führe die Funktion parallel aus
        results = pool.map(game_function, range(num_games))

    # Sammle die Ergebnisse aus allen Prozessen
    all_training_data = []
    for game_data in results:
        all_training_data.extend(game_data)

    return all_training_data

# Wrapper-Funktion, die von `pool.map` aufgerufen wird.
# Sie benötigt ein dummy-Argument (hier 'game_index'), um mit `map` kompatibel zu sein.
def play_game_wrapper(game_index, model, num_simulations):
    return play_game(model, num_simulations)


if __name__ == '__main__':
    # Beispiel für die parallele Generierung von Trainingsdaten
    print("Starte Beispiel für paralleles Selbstspiel...")

    # Wichtig: Setze den Start-Method für Multiprocessing. 'fork' kann bei CUDA zu Problemen führen.
    # 'spawn' ist sicherer und plattformunabhängiger.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    model = ChessModel()

    # Führe 4 Partien parallel aus
    num_parallel_games = 4
    training_data = run_parallel_games(model, num_games=num_parallel_games, num_simulations=50)

    print(f"\nParalleles Selbstspiel beendet. {len(training_data)} Trainingsdatensätze generiert.")

    if training_data:
        print(f"Durchschnittliche Anzahl an Datensätzen pro Partie: {len(training_data) / num_parallel_games:.1f}")
        assert len(training_data) > 0
        print("Sanity-Check erfolgreich!")