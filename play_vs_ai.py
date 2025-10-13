import chess
import torch
import os
from neural_network import ChessModel
from mcts import MCTS

def select_player_color():
    """Lässt den Spieler wählen, welche Farbe er spielen möchte."""
    while True:
        choice = input("Möchtest du Weiß (w) oder Schwarz (s) spielen? ").lower()
        if choice in ['w', 'weiss', 'weiß']:
            return chess.WHITE
        elif choice in ['s', 'schwarz']:
            return chess.BLACK
        else:
            print("Ungültige Eingabe. Bitte wähle 'w' oder 's'.")

def get_player_move(board: chess.Board) -> chess.Move:
    """Fordert den Spieler zur Eingabe eines Zuges auf und validiert ihn."""
    while True:
        uci_move = input("Dein Zug (im UCI-Format, z.B. e2e4): ").strip()
        try:
            move = chess.Move.from_uci(uci_move)
            if move in board.legal_moves:
                return move
            else:
                print("Illegaler Zug. Bitte versuche es erneut.")
        except ValueError:
            print("Ungültiges Format. Bitte gib den Zug im UCI-Format an (z.B. e2e4).")

def play_vs_ai(num_simulations: int = 400):
    """
    Hauptfunktion, die eine interaktive Partie zwischen einem menschlichen
    Spieler und der KI startet.
    """
    # Lade das beste verfügbare Modell
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("Fehler: Kein trainiertes Modell gefunden unter 'models/best_model.pth'.")
        print("Bitte führe zuerst ein Training durch (z.B. mit 'train_fast.py').")
        return

    print("Lade KI-Modell...")
    model = ChessModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialisiere MCTS mit dem geladenen Modell
    mcts = MCTS(model)
    board = chess.Board()

    player_color = select_player_color()

    print("\nPartie beginnt. Viel Erfolg!")
    print(board)

    while not board.is_game_over():
        if board.turn == player_color:
            # Menschlicher Spieler ist am Zug
            move = get_player_move(board)
        else:
            # KI ist am Zug
            print("\nKI denkt nach...")
            move = mcts.find_best_move(board, num_simulations)
            print(f"KI spielt: {move.uci()}")

        board.push(move)
        print("\n" + str(board))

    # Partie-Ende
    outcome = board.outcome()
    print("\n--- Partie beendet ---")
    if outcome.winner is None:
        print("Ergebnis: Unentschieden")
    elif outcome.winner == player_color:
        print("Ergebnis: Du hast gewonnen!")
    else:
        print("Ergebnis: Die KI hat gewonnen.")
    print(f"Grund: {outcome.termination.name.capitalize()}")


if __name__ == "__main__":
    # Eine hohe Anzahl an Simulationen für eine stärkere KI im Spiel
    play_vs_ai(num_simulations=800)