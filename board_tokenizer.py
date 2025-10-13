import chess
import torch

# Definiere die Indizes für die speziellen Tokens und Figuren
# Diese werden für die Embedding-Schicht im Modell benötigt.
# Spezielle Tokens
CLS_TOKEN_IDX = 0
SIDE_TO_MOVE_WHITE_IDX = 1
SIDE_TO_MOVE_BLACK_IDX = 2
CASTLING_WK_IDX = 3
CASTLING_WQ_IDX = 4
CASTLING_BK_IDX = 5
CASTLING_BQ_IDX = 6

# Figuren-Tokens (Startindex nach den speziellen Tokens)
PIECE_START_IDX = 7
PIECE_NAMES = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
PIECE_TO_IDX = {name: i + PIECE_START_IDX for i, name in enumerate(PIECE_NAMES)}

# Positions-Tokens
POSITION_START_IDX = PIECE_START_IDX + len(PIECE_NAMES) # 7 + 12 = 19
NUM_SQUARES = 64

VOCAB_SIZE = POSITION_START_IDX + NUM_SQUARES # 19 + 64 = 83

def tokenize_board(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Konvertiert ein chess.Board-Objekt in eine Sequenz von Token-IDs für den Transformer.

    Die Funktion gibt zwei Tensoren zurück:
    1. `token_ids`: Die Sequenz der Haupt-IDs (Figur, spezielle Tokens).
    2. `position_ids`: Die Sequenz der Positions-IDs für jede Figur.

    Args:
        board (chess.Board): Das zu tokenisierende Schachbrett-Objekt.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Ein Tupel aus token_ids und position_ids.
    """

    # 1. Spezielle Tokens am Anfang
    tokens = [CLS_TOKEN_IDX]

    if board.turn == chess.WHITE:
        tokens.append(SIDE_TO_MOVE_WHITE_IDX)
    else:
        tokens.append(SIDE_TO_MOVE_BLACK_IDX)

    if board.has_kingside_castling_rights(chess.WHITE):
        tokens.append(CASTLING_WK_IDX)
    if board.has_queenside_castling_rights(chess.WHITE):
        tokens.append(CASTLING_WQ_IDX)
    if board.has_kingside_castling_rights(chess.BLACK):
        tokens.append(CASTLING_BK_IDX)
    if board.has_queenside_castling_rights(chess.BLACK):
        tokens.append(CASTLING_BQ_IDX)

    # Die Position der speziellen Tokens ist 0, da sie keine Feldposition haben
    positions = [0] * len(tokens)

    # 2. Figuren-Tokens
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Füge die ID der Figur hinzu
            piece_symbol = piece.symbol()
            tokens.append(PIECE_TO_IDX[piece_symbol])

            # Füge die Positions-ID hinzu
            # Wir addieren den Startindex, um Kollisionen mit den speziellen Positionen zu vermeiden
            positions.append(POSITION_START_IDX + square)

    return torch.tensor(tokens, dtype=torch.long), torch.tensor(positions, dtype=torch.long)


if __name__ == '__main__':
    # Beispiel für die Tokenisierung
    board = chess.Board() # Startposition

    token_ids, position_ids = tokenize_board(board)

    print("--- Tokenizer-Check ---")
    print(f"Board FEN: {board.fen()}")
    print("\nToken IDs:")
    print(token_ids)
    print("\nPosition IDs (0 für spezielle Tokens):")
    print(position_ids)

    # Die Länge sollte 6 (spezielle Tokens) + 32 (Figuren) = 38 sein
    print(f"\nLänge der Token-Sequenz: {len(token_ids)}")
    print(f"Länge der Positions-Sequenz: {len(position_ids)}")
    assert len(token_ids) == 38
    assert len(position_ids) == 38

    # Überprüfe einen spezifischen Token
    # Weißer Bauer auf e2 (square 12)
    # Symbol 'P', Index = 7 + 6 = 13
    # Position = 19 + 12 = 31

    # Finde den Index des Tokens für das Feld e2 auf robuste Weise
    num_special_tokens = 0
    if CLS_TOKEN_IDX is not None: num_special_tokens += 1
    if SIDE_TO_MOVE_WHITE_IDX is not None: num_special_tokens += 1 # Side to move
    if CASTLING_WK_IDX is not None: num_special_tokens += board.has_castling_rights(chess.WHITE) # All castling rights

    pieces_before_e2 = 0
    for i in range(12): # Squares 0 (a1) to 11 (d2)
        if board.piece_at(i):
            pieces_before_e2 += 1

    # Der Index in der Sequenz ist die Anzahl der speziellen Tokens plus die Anzahl der Figuren vor e2
    pawn_e2_idx_in_seq = len([t for t in [CLS_TOKEN_IDX, SIDE_TO_MOVE_WHITE_IDX, CASTLING_WK_IDX, CASTLING_WQ_IDX, CASTLING_BK_IDX, CASTLING_BQ_IDX] if t is not None and (t < PIECE_START_IDX or board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK))]) - 4 + sum(1 for i in range(12) if board.piece_at(i))

    # Simplere, aber korrekte Logik, da wir die Anzahl der Tokens am Anfang kennen
    # 6 spezielle Tokens + 8 Figuren auf der ersten Reihe + 4 Bauern (a2-d2)
    num_special_tokens = 6 # In der Startaufstellung
    pawn_e2_idx_in_seq = num_special_tokens + 8 + 4

    assert token_ids[pawn_e2_idx_in_seq] == PIECE_TO_IDX['P']
    assert position_ids[pawn_e2_idx_in_seq] == POSITION_START_IDX + 12

    print("\nSanity-Checks erfolgreich!")