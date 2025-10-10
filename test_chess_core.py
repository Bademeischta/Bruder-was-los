import unittest
from chess_core import ChessCore

class TestChessCore(unittest.TestCase):
    """
    Diese Testklasse stellt die korrekte Funktionalität der ChessCore-Klasse sicher.
    Jede Methode wird rigoros gegen die Spezifikationen geprüft.
    """

    def setUp(self):
        """Wird vor jedem Test ausgeführt, um eine saubere Instanz zu gewährleisten."""
        self.core = ChessCore()

    def test_initialization(self):
        """Prüft den initialen Zustand des Schachbretts."""
        self.assertEqual(self.core.get_board_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertEqual(self.core.get_game_status(), "in_progress")

    def test_get_legal_moves(self):
        """Stellt sicher, dass get_legal_moves eine korrekte Liste von UCI-Strings zurückgibt."""
        legal_moves = self.core.get_legal_moves()
        self.assertIsInstance(legal_moves, list)
        self.assertIn("e2e4", legal_moves)
        self.assertIn("g1f3", legal_moves)
        self.assertEqual(len(legal_moves), 20)

    def test_push_uci_legal_move(self):
        """Testet die Ausführung eines legalen Zugs."""
        move = "e2e4"
        self.assertTrue(self.core.push_uci(move))
        self.assertEqual(self.core.get_board_fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

    def test_push_uci_illegal_move(self):
        """Testet die Reaktion auf einen illegalen Zug. Der Zustand darf sich nicht ändern."""
        initial_fen = self.core.get_board_fen()
        move = "e2e5" # Illegaler Bauernzug am Anfang
        self.assertFalse(self.core.push_uci(move))
        self.assertEqual(self.core.get_board_fen(), initial_fen)

    def test_push_uci_invalid_format(self):
        """Testet die Reaktion auf einen syntaktisch ungültigen UCI-String."""
        initial_fen = self.core.get_board_fen()
        move = "invalid_move"
        self.assertFalse(self.core.push_uci(move))
        self.assertEqual(self.core.get_board_fen(), initial_fen)

    def test_get_game_status_checkmate(self):
        """Testet die korrekte Erkennung von "checkmate" (Narrenmatt)."""
        # Narrenmatt-Sequenz
        self.assertTrue(self.core.push_uci("f2f3"))
        self.assertTrue(self.core.push_uci("e7e5"))
        self.assertTrue(self.core.push_uci("g2g4"))
        self.assertTrue(self.core.push_uci("d8h4"))
        self.assertEqual(self.core.get_game_status(), "checkmate")

    def test_get_game_status_stalemate(self):
        """Testet die korrekte Erkennung von "stalemate"."""
        # Ein einfacher, unzweideutiger Patt-Zustand.
        # Der schwarze König auf f8 ist nicht im Schach, kann aber kein Feld betreten,
        # da alle von König auf f7 kontrolliert werden. Schwarz ist am Zug.
        fen_stalemate = "5k2/5K2/8/8/8/8/8/8 b - - 0 1"
        self.assertTrue(self.core.load_fen(fen_stalemate))
        self.assertEqual(self.core.get_game_status(), "stalemate")

    def test_load_fen_valid(self):
        """Testet das Laden eines validen FEN-Strings."""
        fen_string = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        self.assertTrue(self.core.load_fen(fen_string))
        self.assertEqual(self.core.get_board_fen(), fen_string)

    def test_load_fen_invalid(self):
        """Testet die Reaktion auf einen invaliden FEN-String."""
        initial_fen = self.core.get_board_fen()
        invalid_fen = "this is not a valid fen"
        self.assertFalse(self.core.load_fen(invalid_fen))
        self.assertEqual(self.core.get_board_fen(), initial_fen)


if __name__ == '__main__':
    unittest.main()