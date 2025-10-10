import chess

# Die Policy-Repräsentation ist ein 8x8x73 Tensor.
# 73 Ebenen für jeden der 64 Startquadrate.
# Ebenen 0-55: "Queen"-Züge (8 Richtungen, 1-7 Felder weit)
# Ebenen 56-63: Springerzüge (8 mögliche Züge)
# Ebenen 64-72: Bauern-Unterverwandlungen (3 Richtungen * 3 Figuren)

# Richtungsvektoren (dr, dc) für die 8 Queen-Richtungen
QUEEN_DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
# (dr, dc) für die 8 Springer-Richtungen
KNIGHT_DIRECTIONS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
# Mapping von Unterverwandlungsfigur zu Index
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Konvertiert ein chess.Move-Objekt in seinen entsprechenden Index (0-4671).
    """
    from_square = move.from_square
    to_square = move.to_square
    from_rank, from_file = chess.square_rank(from_square), chess.square_file(from_square)
    to_rank, to_file = chess.square_rank(to_square), chess.square_file(to_square)

    dr, dc = to_rank - from_rank, to_file - from_file

    # Fall 1: Unterverwandlung eines Bauern
    if move.promotion and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        # dc ist -1 (links), 0 (vorwärts), 1 (rechts)
        direction_idx = dc + 1
        plane_index = 64 + direction_idx * 3 + piece_idx
    else:
        # Fall 2: Springerzug
        is_knight_move = (abs(dr) == 2 and abs(dc) == 1) or \
                         (abs(dr) == 1 and abs(dc) == 2)
        if is_knight_move:
            knight_idx = KNIGHT_DIRECTIONS.index((dr, dc))
            plane_index = 56 + knight_idx
        else:
            # Fall 3: "Queen"-Zug (inkl. Bauern-, Königs-, Turm-, Läufer- und Damen-Züge)
            distance = max(abs(dr), abs(dc))
            direction_dr = dr // distance if distance != 0 else 0
            direction_dc = dc // distance if distance != 0 else 0

            direction_idx = QUEEN_DIRECTIONS.index((direction_dr, direction_dc))
            plane_index = direction_idx * 7 + (distance - 1)

    square_index = from_rank * 8 + from_file
    return square_index * 73 + plane_index

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Konvertiert einen Index (0-4671) zurück in ein chess.Move-Objekt.
    Diese Funktion garantiert nicht, dass der Zug legal ist.
    """
    plane_index = index % 73
    square_index = index // 73

    from_rank = square_index // 8
    from_file = square_index % 8
    from_square = chess.square(from_file, from_rank)

    promotion = None

    if plane_index < 56: # Queen-Züge
        direction_idx = plane_index // 7
        distance = (plane_index % 7) + 1
        dr, dc = QUEEN_DIRECTIONS[direction_idx]
        to_rank, to_file = from_rank + dr * distance, from_file + dc * distance

        # Prüfen auf Bauernverwandlung zur Dame
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and from_rank == 6 and to_rank == 7) or \
               (piece.color == chess.BLACK and from_rank == 1 and to_rank == 0):
                promotion = chess.QUEEN
    elif plane_index < 64: # Springer-Züge
        knight_idx = plane_index - 56
        dr, dc = KNIGHT_DIRECTIONS[knight_idx]
        to_rank, to_file = from_rank + dr, from_file + dc
    else: # Unterverwandlungen
        underpromotion_idx = plane_index - 64
        direction_idx = underpromotion_idx // 3
        piece_idx = underpromotion_idx % 3

        promotion = UNDERPROMOTION_PIECES[piece_idx]
        dc = direction_idx - 1

        # Bestimme die Richtung basierend auf der Farbe des Spielers am Zug
        dr = 1 if board.turn == chess.WHITE else -1
        to_rank, to_file = from_rank + dr, from_file + dc

    # Stellen Sie sicher, dass die Zielkoordinaten auf dem Brett liegen
    if 0 <= to_rank <= 7 and 0 <= to_file <= 7:
        to_square = chess.square(to_file, to_rank)
        return chess.Move(from_square, to_square, promotion)

    return None # Sollte selten passieren, wenn die Logik stimmt