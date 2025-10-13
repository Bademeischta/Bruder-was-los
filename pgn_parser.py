import chess
import chess.pgn
import numpy as np
from board_tokenizer import tokenize_board
from move_translator import move_to_index

def parse_pgn_file(pgn_filepath: str) -> list:
    """
    Liest eine PGN-Datei, durchläuft alle Partien und konvertiert sie in
    ein für das Training geeignetes Format.

    Args:
        pgn_filepath (str): Der Pfad zur PGN-Datei.

    Returns:
        list: Eine Liste von Trainings-Tupeln:
              [(token_ids, position_ids, policy_vector, outcome), ...].
    """
    training_data = []

    with open(pgn_filepath) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # Ende der Datei

            # Bestimme den Ausgang der Partie
            result = game.headers.get("Result")
            if result == "1-0":
                game_outcome = 1.0  # Weiß gewinnt
            elif result == "0-1":
                game_outcome = -1.0 # Schwarz gewinnt
            else:
                game_outcome = 0.0  # Unentschieden

            board = game.board()
            game_history = []

            # Durchlaufe alle Züge der Partie
            for move in game.mainline_moves():
                # Tokenisiere den Zustand *vor* dem Zug
                token_ids, position_ids = tokenize_board(board)

                # Erstelle den Policy-Vektor: 1.0 für den gespielten Zug, 0.0 sonst
                policy_vector = np.zeros(4672, dtype=np.float32)
                move_idx = move_to_index(move, board)
                policy_vector[move_idx] = 1.0

                # Speichere die Daten für diesen Zug
                game_history.append([token_ids, position_ids, policy_vector, 0.0]) # Outcome wird später gesetzt

                board.push(move)

            # Trage den finalen Ausgang in alle Trainingsdaten dieser Partie ein
            current_player = chess.WHITE
            for i in range(len(game_history)):
                if current_player == chess.WHITE:
                    game_history[i][3] = game_outcome
                else:
                    game_history[i][3] = -game_outcome
                current_player = not current_player

            training_data.extend(game_history)

    return training_data

if __name__ == '__main__':
    # Beispiel für die Nutzung des PGN-Parsers
    # Erstelle eine Dummy-PGN-Datei für Testzwecke
    dummy_pgn_content = """
[Event "Fictional Game"]
[Site "?"]
[Date "2023.10.13"]
[Round "?"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""
    dummy_pgn_filepath = "dummy_game.pgn"
    with open(dummy_pgn_filepath, "w") as f:
        f.write(dummy_pgn_content)

    print(f"Lese und parse die Dummy-PGN-Datei: {dummy_pgn_filepath}")

    parsed_data = parse_pgn_file(dummy_pgn_filepath)

    print(f"\n{len(parsed_data)} Trainingsdatensätze wurden aus der PGN-Datei extrahiert.")

    if parsed_data:
        first_state_tokens, first_state_pos, first_policy, first_outcome = parsed_data[0]

        print("\n--- Erster Datensatz (Stellung vor 1. e4) ---")
        print(f"Token IDs Shape: {first_state_tokens.shape}")
        print(f"Position IDs Shape: {first_state_pos.shape}")
        print(f"Policy Vector Shape: {first_policy.shape}")
        print(f"Summe der Policy-Wahrscheinlichkeiten: {np.sum(first_policy)}")
        print(f"Partienausgang (aus Perspektive von Weiß): {first_outcome}")

        # Der erste Zug war e2e4. Der Index dafür ist 877.
        # Der Policy-Vektor sollte an dieser Stelle eine 1 haben.
        assert np.isclose(np.sum(first_policy), 1.0)
        assert first_policy[877] == 1.0
        assert first_outcome == 1.0

        print("\nSanity-Checks erfolgreich!")

    # Aufräumen
    import os
    os.remove(dummy_pgn_filepath)