import unittest
import chess
from move_translator import move_to_index, index_to_move

class TestMoveTranslator(unittest.TestCase):
    """
    Testet die bidirektionale Übersetzung zwischen chess.Move-Objekten
    und den Policy-Indizes (0-4671).
    """

    def test_simple_pawn_move(self):
        """Testet einen einfachen Bauernzug."""
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        index = move_to_index(move, board)

        # Manuelle Berechnung zur Überprüfung:
        # from_square e2 = 12, plane für (dr=2, dc=0) -> (1,0) dir -> 0, dist 2 -> 0*7 + 1 = 1
        # 12 * 73 + 1 = 877
        self.assertEqual(index, 877)

        converted_move = index_to_move(index, board)
        self.assertEqual(move, converted_move)

    def test_knight_move(self):
        """Testet einen Springerzug."""
        board = chess.Board()
        move = chess.Move.from_uci("g1f3")
        index = move_to_index(move, board)

        # Korrekte Berechnung:
        # from_square g1=6. dr=2, dc=-1 -> knight_dir_idx = 7. plane = 56+7=63.
        # index = 6 * 73 + 63 = 501.
        self.assertEqual(index, 501)

        converted_move = index_to_move(index, board)
        self.assertEqual(move, converted_move)

    def test_queen_side_castle(self):
        """Testet einen Rochadezug (wird als Königszug kodiert)."""
        # Rochade ist nur ein Königszug um 2 Felder
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1c1")
        index = move_to_index(move, board)
        converted_move = index_to_move(index, board)
        self.assertEqual(move, converted_move)

    def test_promotion_to_queen(self):
        """Testet eine Bauernumwandlung zur Dame."""
        # Brett mit weißem Bauer auf b7, bereit zur Umwandlung
        board = chess.Board("k7/1P6/8/8/8/8/8/K7 w - - 0 1")

        move = chess.Move.from_uci("b7a8q") # Umwandlung zur Dame
        index = move_to_index(move, board)
        converted_move = index_to_move(index, board)
        self.assertEqual(move, converted_move)

    def test_underpromotion_to_rook(self):
        """Testet eine Unterverwandlung zum Turm."""
        # Brett mit weißem Bauer auf b7, bereit zur Umwandlung
        board = chess.Board("k7/1P6/8/8/8/8/8/K7 w - - 0 1")

        move = chess.Move.from_uci("b7a8r") # Unterverwandlung zum Turm
        index = move_to_index(move, board)
        converted_move = index_to_move(index, board)
        self.assertEqual(move, converted_move)

    def test_all_legal_moves_bidirectional(self):
        """
        Testet für eine komplexe Stellung, ob alle legalen Züge korrekt
        hin- und zurückkonvertiert werden können.
        """
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)

        for move in board.legal_moves:
            with self.subTest(move=move.uci()):
                index = move_to_index(move, board)
                converted_move = index_to_move(index, board)
                self.assertEqual(move, converted_move)

if __name__ == '__main__':
    unittest.main()