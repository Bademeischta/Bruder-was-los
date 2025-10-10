import chess
import numpy as np

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Konvertiert ein `chess.Board`-Objekt in einen 21x8x8 Tensor,
    der die in ARCHITECTURE.md definierte Datenrepräsentation widerspiegelt.

    Args:
        board (chess.Board): Das zu konvertierende Schachbrett-Objekt.

    Returns:
        np.ndarray: Ein Tensor der Dimension (21, 8, 8) als float32.
    """
    # Tensor-Dimension: (21, 8, 8)
    tensor = np.zeros((21, 8, 8), dtype=np.float32)

    # Mapping von Figurentyp zu Ebenen-Index (0-5)
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # 1. Figurenpositionen (12 Kanäle)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Rank und File in Matrix-Indizes umwandeln (0-7)
            rank, file = chess.square_rank(square), chess.square_file(square)

            plane_idx = piece_to_plane[piece.piece_type]
            if piece.color != board.turn:
                # Figuren des Gegners kommen auf die Ebenen 6-11
                plane_idx += 6

            tensor[plane_idx, rank, file] = 1

    # 2. Spielzustand / Meta-Informationen (7 Kanäle)
    # Rochaderechte
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1

    # Spieler am Zug
    if board.turn == chess.WHITE:
        tensor[16, :, :] = 1

    # Gesamtzahl der Züge (normalisiert)
    tensor[17, :, :] = board.fullmove_number / 200.0 # Normalisierungsfaktor 200

    # 50-Züge-Regel-Zähler (normalisiert)
    tensor[18, :, :] = board.halfmove_clock / 100.0 # Normalisierungsfaktor 100

    # 3. Zugwiederholung (2 Kanäle)
    # `is_repetition(2)` prüft, ob die Position zum 2. Mal vorkommt (also 1 Wiederholung)
    if board.is_repetition(2) and not board.is_repetition(3):
        tensor[19, :, :] = 1
    # `is_repetition(3)` prüft, ob die Position zum 3. Mal vorkommt (also 2 Wiederholungen)
    if board.is_repetition(3):
         tensor[20, :, :] = 1

    return tensor

if __name__ == '__main__':
    # Beispiel für die Konvertierung und einen Sanity-Check
    board = chess.Board() # Startposition

    # Mache ein paar Züge, um den Zustand zu ändern
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    board.push_uci("g1f3")
    board.push_uci("b8c6")

    # Konvertiere das Brett in einen Tensor
    tensor_representation = board_to_tensor(board)

    print("--- Tensor-Konvertierung-Check ---")
    print(f"Board FEN: {board.fen()}")
    print(f"Tensor Shape: {tensor_representation.shape}")
    assert tensor_representation.shape == (21, 8, 8)

    # Prüfe einige erwartete Werte
    # Weißer Bauer auf e4 (Rank 3, File 4)
    # Weiß ist am Zug, also ist es eine Figur des aktuellen Spielers (Ebene 0)
    assert tensor_representation[0, 3, 4] == 1
    # Schwarzer Bauer auf e5 (Rank 4, File 4)
    # Weiß ist am Zug, also ist es eine Figur des Gegners (Ebene 6)
    assert tensor_representation[6, 4, 4] == 1
    # Weißer Springer auf f3 (Rank 2, File 5)
    assert tensor_representation[1, 2, 5] == 1
    # Schwarzer Springer auf c6 (Rank 5, File 2)
    assert tensor_representation[7, 5, 2] == 1
    # Weiß ist am Zug
    assert tensor_representation[16, 0, 0] == 1
    # Rochaderechte sollten noch bestehen
    assert tensor_representation[12, 0, 0] == 1
    assert tensor_representation[13, 0, 0] == 1
    assert tensor_representation[14, 0, 0] == 1
    assert tensor_representation[15, 0, 0] == 1

    print("\nSanity-Check der Tensor-Werte erfolgreich!")