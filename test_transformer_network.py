import unittest
import torch
import chess
from transformer_network import ChessTransformer
from board_tokenizer import tokenize_board

class TestTransformerNetwork(unittest.TestCase):
    """
    Testet die Transformer-Netzwerkarchitektur.
    """

    def test_forward_pass_single_item(self):
        """Testet einen Forward-Pass mit einer einzelnen Brettstellung."""
        model = ChessTransformer()
        model.eval()

        board = chess.Board()
        token_ids, position_ids = tokenize_board(board)

        # Füge eine Batch-Dimension hinzu
        token_ids = token_ids.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = model(token_ids, position_ids)

        # Erwartete Dimensionen für einen Batch der Größe 1
        # Policy: (1, NumTokens - 1, 73)
        # Value: (1, 1)
        self.assertEqual(policy_logits.shape, (1, token_ids.shape[1] - 1, 73))
        self.assertEqual(value.shape, (1, 1))

        # Der Value sollte zwischen -1 und 1 liegen
        self.assertTrue(-1 <= value.item() <= 1)

    def test_forward_pass_batch_with_padding(self):
        """
        Testet einen Forward-Pass mit einem Batch von Stellungen unterschiedlicher
        Länge (unterschiedliche Anzahl an Figuren), was Padding erfordert.
        """
        from torch.nn.utils.rnn import pad_sequence

        model = ChessTransformer()
        model.eval()

        # Erstelle zwei Stellungen mit unterschiedlicher Figurenanzahl
        board1 = chess.Board() # 32 Figuren
        board2 = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2") # 32 Figuren
        board3 = chess.Board("8/8/8/4k3/8/8/8/R3K3 w Q - 0 1") # 4 Figuren

        # Tokenisiere die Stellungen
        t1, p1 = tokenize_board(board1)
        t2, p2 = tokenize_board(board2)
        t3, p3 = tokenize_board(board3)

        # Erstelle den Batch und das Padding
        token_batch = [t1, t2, t3]
        pos_batch = [p1, p2, p3]

        tokens_padded = pad_sequence(token_batch, batch_first=True, padding_value=0)
        pos_padded = pad_sequence(pos_batch, batch_first=True, padding_value=0)
        padding_mask = (tokens_padded == 0)

        with torch.no_grad():
            policy_logits, value = model(tokens_padded, pos_padded, src_key_padding_mask=padding_mask)

        # Erwartete Dimensionen für einen Batch der Größe 3
        # Policy: (3, MaxSeqLen - 1, 73)
        # Value: (3, 1)
        self.assertEqual(policy_logits.shape, (3, tokens_padded.shape[1] - 1, 73))
        self.assertEqual(value.shape, (3, 1))

if __name__ == '__main__':
    unittest.main()