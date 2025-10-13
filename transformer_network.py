import torch
import torch.nn as nn
import math
from board_tokenizer import VOCAB_SIZE, PIECE_START_IDX, NUM_SQUARES

class ChessTransformer(nn.Module):
    """
    Eine Transformer-basierte Netzwerkarchitektur für Schach, die darauf ausgelegt ist,
    Beziehungen zwischen Figuren zu lernen, anstatt nur räumliche Muster.
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(ChessTransformer, self).__init__()
        self.d_model = d_model

        # Embedding-Schichten
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        # Das Positions-Embedding muss alle möglichen Positions-IDs abdecken können,
        # die vom Tokenizer generiert werden. Die höchste ID ist VOCAB_SIZE - 1.
        self.position_embedding = nn.Embedding(VOCAB_SIZE, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Output Heads
        # Value Head: Nimmt die Ausgabe des [CLS]-Tokens
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )

        # Policy Head: Sagt für jede Figur die möglichen Aktionen voraus
        # Die Ausgabe wird später zu dem 4672-Vektor zusammengesetzt
        self.policy_head = nn.Linear(d_model, 73) # 73 mögliche Aktionstypen pro Figur

    def forward(self, token_ids: torch.Tensor, position_ids: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Führt den Forward-Pass durch das Transformer-Modell aus.

        Args:
            token_ids (torch.Tensor): Die Sequenz der Haupt-Token-IDs (Batch, SeqLen).
            position_ids (torch.Tensor): Die Sequenz der Positions-IDs (Batch, SeqLen).
            src_key_padding_mask (torch.Tensor): Eine Maske, um Padding-Tokens in der
                                                 Sequenz zu ignorieren.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Ein Tupel aus:
                - policy_logits (Batch, NumPieces, 73): Die "lokalen" Logits für jede Figur.
                - value (Batch, 1): Die Bewertung der Position.
        """
        # 1. Embeddings erstellen
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(position_ids)
        src = token_emb + pos_emb # Kombiniere die Embeddings

        # 2. Transformer Encoder
        transformer_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # 3. Value Head
        # Wir verwenden die Ausgabe des ersten Tokens ([CLS]-Token) für die globale Bewertung
        cls_token_output = transformer_output[:, 0, :]
        value = self.value_head(cls_token_output)

        # 4. Policy Head
        # Wir verwenden die Ausgaben der restlichen Tokens (Figuren-Tokens)
        piece_tokens_output = transformer_output[:, 1:, :] # Ignoriere [CLS]
        policy_logits = self.policy_head(piece_tokens_output)

        return policy_logits, value

if __name__ == '__main__':
    # Beispiel für die Instanziierung und einen Forward-Pass
    from board_tokenizer import tokenize_board
    import chess

    model = ChessTransformer()
    board = chess.Board()

    # Tokenisiere das Brett
    tokens, positions = tokenize_board(board)
    # Füge eine Batch-Dimension hinzu
    tokens = tokens.unsqueeze(0)
    positions = positions.unsqueeze(0)

    print("--- Transformer-Check ---")
    print(f"Input Token Shape:  {tokens.shape}")
    print(f"Input Position Shape: {positions.shape}")

    # Führe den Forward-Pass aus
    policy, value = model(tokens, positions)

    print(f"\nPolicy Logits Shape: {policy.shape}") # Erwartet: (1, 38-1, 73) -> (1, 37, 73)
    print(f"Value Shape: {value.shape}")         # Erwartet: (1, 1)

    # In der Startaufstellung sind 32 Figuren, also 32 Piece-Tokens
    # Plus 6 spezielle Tokens am Anfang. Die Sequenzlänge ist 38.
    # Policy wird auf die 37 Tokens nach [CLS] angewendet.
    # In einem Batch würde man Padding benötigen.
    num_special_tokens = 6
    num_pieces = 32
    assert policy.shape == (1, num_special_tokens + num_pieces - 1, 73)
    assert value.shape == (1, 1)
    print("\nDimensions-Check erfolgreich!")