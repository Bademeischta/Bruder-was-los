import unittest
import chess
import torch
from board_tokenizer import tokenize_board, PIECE_TO_IDX, POSITION_START_IDX

class TestBoardTokenizer(unittest.TestCase):
    """Testet die Tokenisierungslogik für den Transformer."""

    def test_initial_position(self):
        """Testet die Tokenisierung der Startposition."""
        board = chess.Board()
        token_ids, position_ids = tokenize_board(board)

        # Erwartete Länge: 6 spezielle Tokens + 32 Figuren
        self.assertEqual(token_ids.shape, (38,))
        self.assertEqual(position_ids.shape, (38,))

        # Überprüfe den [CLS] Token
        self.assertEqual(token_ids[0], 0)
        self.assertEqual(position_ids[0], 0)

        # Überprüfe den "Side to Move" Token
        self.assertEqual(token_ids[1], 1) # SIDE_TO_MOVE_WHITE_IDX

        # Überprüfe die Rochaderechte
        self.assertIn(3, token_ids) # WK
        self.assertIn(4, token_ids) # WQ
        self.assertIn(5, token_ids) # BK
        self.assertIn(6, token_ids) # BQ

    def test_specific_piece_tokenization(self):
        """Testet die Tokenisierung einer spezifischen Figur."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        token_ids, position_ids = tokenize_board(board)

        # Finde den Token für den König auf e1 (square 4)
        target_pos_id = POSITION_START_IDX + 4

        # Finde den Index in der Sequenz, wo die Positions-ID übereinstimmt
        seq_idx = (position_ids == target_pos_id).nonzero(as_tuple=True)[0].item()

        # Überprüfe, ob der Token-ID an dieser Stelle dem weißen König entspricht
        self.assertEqual(token_ids[seq_idx], PIECE_TO_IDX['K'])

if __name__ == '__main__':
    unittest.main()