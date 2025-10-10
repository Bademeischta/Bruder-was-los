import unittest
import chess
import numpy as np
from state_encoder import board_to_tensor

class TestStateEncoder(unittest.TestCase):
    """
    Testet die `board_to_tensor`-Funktion rigoros, um die korrekte
    Konvertierung von FEN-Zuständen in die Tensor-Repräsentation sicherzustellen.
    """

    def test_initial_position(self):
        """Testet die FEN-Startposition."""
        board = chess.Board()
        tensor = board_to_tensor(board)

        self.assertEqual(tensor.shape, (21, 8, 8))

        # Test: Weiße Bauern in Reihe 2 (Index 1) auf Ebene 0
        self.assertTrue(np.all(tensor[0, 1, :] == 1))
        # Test: Schwarze Bauern in Reihe 7 (Index 6) auf Ebene 6
        self.assertTrue(np.all(tensor[6, 6, :] == 1))
        # Test: Weißer Turm auf a1 (0,0) auf Ebene 3
        self.assertEqual(tensor[3, 0, 0], 1)
        # Test: Schwarzer Springer auf g8 (7,6) auf Ebene 7
        self.assertEqual(tensor[7, 7, 6], 1)

        # Test: Alle Rochaderechte sind vorhanden
        self.assertTrue(np.all(tensor[12, :, :] == 1)) # WK
        self.assertTrue(np.all(tensor[13, :, :] == 1)) # WQ
        self.assertTrue(np.all(tensor[14, :, :] == 1)) # BK
        self.assertTrue(np.all(tensor[15, :, :] == 1)) # BQ

        # Test: Weiß ist am Zug
        self.assertTrue(np.all(tensor[16, :, :] == 1))

        # Test: Keine Wiederholungen
        self.assertTrue(np.all(tensor[19, :, :] == 0))
        self.assertTrue(np.all(tensor[20, :, :] == 0))

    def test_sicilian_defense_black_to_move(self):
        """Testet eine typische Mittelspielstellung (Sizilianisch), Schwarz am Zug."""
        fen = "r1bqkbnr/pp1ppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
        board = chess.Board(fen)
        tensor = board_to_tensor(board)

        # Test: Weißer Bauer auf e4 (3,4) ist nun eine gegnerische Figur
        self.assertEqual(tensor[6, 3, 4], 1) # Ebene 6 = Gegnerischer Bauer
        # Test: Schwarzer Springer auf c6 (5,2) ist eine Figur des aktuellen Spielers
        self.assertEqual(tensor[1, 5, 2], 1) # Ebene 1 = Eigener Springer

        # Test: Schwarz ist am Zug
        self.assertTrue(np.all(tensor[16, :, :] == 0))

        # Test: Rochaderechte sind unverändert
        self.assertTrue(np.all(tensor[12, :, :] == 1))
        self.assertTrue(np.all(tensor[13, :, :] == 1))

    def test_no_castling_rights(self):
        """Testet eine Stellung, in der alle Rochaderechte verloren gegangen sind."""
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
        board = chess.Board(fen)
        # Könige und Türme bewegen, um Rechte zu verlieren
        board.push_uci("e1e2")
        board.push_uci("e8e7")
        tensor = board_to_tensor(board)

        # Test: Alle Rochaderechte sind 0
        self.assertTrue(np.all(tensor[12, :, :] == 0))
        self.assertTrue(np.all(tensor[13, :, :] == 0))
        self.assertTrue(np.all(tensor[14, :, :] == 0))
        self.assertTrue(np.all(tensor[15, :, :] == 0))

    def test_repetition_planes(self):
        """Testet die Ebenen für Zugwiederholungen."""
        board = chess.Board()
        # Eine einfache Zugfolge, die zu einer Wiederholung führt
        board.push_uci("g1f3")
        board.push_uci("g8f6")
        board.push_uci("f3g1")
        board.push_uci("f6g8") # Position ist jetzt zum 2. Mal da (1. Wiederholung)

        tensor1 = board_to_tensor(board)
        self.assertTrue(np.all(tensor1[19, :, :] == 1), "Ebene für 1. Wiederholung sollte 1 sein")
        self.assertTrue(np.all(tensor1[20, :, :] == 0), "Ebene für 2. Wiederholung sollte 0 sein")

        board.push_uci("g1f3")
        board.push_uci("g8f6")
        board.push_uci("f3g1")
        board.push_uci("f6g8") # Position ist jetzt zum 3. Mal da (2. Wiederholung)

        tensor2 = board_to_tensor(board)
        # is_repetition(3) ist jetzt True
        self.assertTrue(np.all(tensor2[19, :, :] == 0), "Ebene für 1. Wiederholung sollte wieder 0 sein")
        self.assertTrue(np.all(tensor2[20, :, :] == 1), "Ebene für 2. Wiederholung sollte 1 sein")


if __name__ == '__main__':
    unittest.main()