import chess
from chess_core import ChessCore
from neural_network import ChessModel
from mcts import MCTS

def main():
    """
    Hauptfunktion, die alle KI-Komponenten integriert und den besten Zug
    für die aktuelle Brettstellung findet.
    """
    print("Initialisiere KI-Komponenten...")

    # 1. Initialisiere das Schachspiel
    core = ChessCore()
    board = chess.Board(core.get_board_fen())

    # 2. Lade das neuronale Netzwerk
    # In einem echten Szenario würden hier trainierte Gewichte geladen
    model = ChessModel()

    # 3. Initialisiere die MCTS-Suche
    mcts = MCTS(model)

    # 4. Führe die Suche aus
    num_simulations = 50 # Eine kleine Anzahl für einen schnellen Test
    print(f"Führe MCTS mit {num_simulations} Simulationen für die Startposition aus...")

    best_move = mcts.find_best_move(board, num_simulations)

    # 5. Gib das Ergebnis aus
    if best_move:
        print(f"\nBester gefundener Zug: {best_move.uci()}")
    else:
        print("\nKein Zug gefunden.")

if __name__ == "__main__":
    main()