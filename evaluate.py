import chess
from neural_network import ChessModel
from mcts import MCTS

def play_match(model1: ChessModel, model2: ChessModel, num_simulations: int) -> int:
    """
    Spielt eine einzelne Partie zwischen zwei Modellen.
    Modell 1 spielt immer Weiß.

    Args:
        model1 (ChessModel): Das erste Modell (spielt Weiß).
        model2 (ChessModel): Das zweite Modell (spielt Schwarz).
        num_simulations (int): Die Anzahl der MCTS-Simulationen pro Zug.

    Returns:
        int: 1 wenn Modell 1 gewinnt, -1 wenn Modell 2 gewinnt, 0 bei Unentschieden.
    """
    board = chess.Board()
    mcts1 = MCTS(model1)
    mcts2 = MCTS(model2)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = mcts1.find_best_move(board, num_simulations)
        else:
            move = mcts2.find_best_move(board, num_simulations)

        if move is None: # Sollte nicht passieren, wenn Züge verfügbar sind
            break

        board.push(move)

    outcome = board.outcome()
    if outcome is None:
        return 0
    if outcome.winner == chess.WHITE:
        return 1
    if outcome.winner == chess.BLACK:
        return -1
    return 0

def run_tournament(new_model: ChessModel, best_model: ChessModel, num_matches: int, num_simulations: int) -> float:
    """
    Führt ein Turnier zwischen einem neuen und dem besten Modell durch.

    Args:
        new_model (ChessModel): Das neu trainierte Modell.
        best_model (ChessModel): Das aktuell beste Modell.
        num_matches (int): Die Anzahl der zu spielenden Partien (muss gerade sein).
        num_simulations (int): Die Anzahl der MCTS-Simulationen pro Zug.

    Returns:
        float: Die Gewinnrate des neuen Modells.
    """
    if num_matches % 2 != 0:
        raise ValueError("Die Anzahl der Matches muss gerade sein, um faire Farben zu gewährleisten.")

    new_model_wins = 0

    for i in range(num_matches // 2):
        print(f"Spiele Match {i*2+1}/{num_matches} (Neues Modell spielt Weiß)")
        # Neues Modell spielt Weiß
        result1 = play_match(new_model, best_model, num_simulations)
        if result1 == 1:
            new_model_wins += 1

        print(f"Spiele Match {i*2+2}/{num_matches} (Neues Modell spielt Schwarz)")
        # Neues Modell spielt Schwarz
        result2 = play_match(best_model, new_model, num_simulations)
        if result2 == -1:
            new_model_wins += 1

    return new_model_wins / num_matches

if __name__ == '__main__':
    print("Starte Beispiel für Evaluations-Modul...")

    # Erzeuge zwei untrainierte Modelle
    model_v1 = ChessModel()
    model_v2 = ChessModel()

    # Führe ein kurzes Turnier durch
    num_games = 4
    sims_per_move = 10
    print(f"Führe ein Turnier mit {num_games} Partien durch...")

    win_rate_v2 = run_tournament(model_v2, model_v1, num_matches=num_games, num_simulations=sims_per_move)

    print(f"\nTurnier beendet.")
    print(f"Gewinnrate von Modell V2 gegen V1: {win_rate_v2 * 100:.1f}%")

    # Akzeptanzkriterium-Check
    if win_rate_v2 > 0.55:
        print("Modell V2 wird als neues bestes Modell akzeptiert.")
    else:
        print("Modell V2 hat das Akzeptanzkriterium nicht erfüllt.")