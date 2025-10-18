import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from transformer_network import ChessTransformer
from board_tokenizer import POSITION_START_IDX

# Die Gesamtanzahl der möglichen Züge in der AlphaZero-Policy-Repräsentation
POLICY_VECTOR_SIZE = 4672

def map_local_to_global_policy(local_policies: torch.Tensor, position_ids: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Mappt die "lokalen" Policy-Logits des Transformers auf den vollständigen
    4672-dimensionalen globalen Policy-Vektor.

    Args:
        local_policies (torch.Tensor): Die Ausgabe des Policy-Heads mit der
                                       Form (Batch, NumPieces, 73).
        position_ids (torch.Tensor): Die Positions-IDs der Tokens mit der
                                     Form (Batch, SeqLen).
        device (str): Das Gerät ('cpu' oder 'cuda'), auf dem die Tensoren erstellt werden sollen.

    Returns:
        torch.Tensor: Ein Tensor mit den globalen Policy-Logits der
                      Form (Batch, 4672).
    """
    batch_size = local_policies.shape[0]
    # Initialisiere mit einem sehr kleinen Wert, um sicherzustellen, dass nicht zugeordnete
    # Logits nach dem Softmax zu ~0 werden.
    global_policy = torch.full((batch_size, POLICY_VECTOR_SIZE), -1e9, device=device)

    # Die lokalen Policies korrespondieren zu den Transformer-Ausgaben ab dem 2. Token ([CLS] wird ignoriert).
    # Daher müssen wir auch die Positions-IDs ab dem 2. Token betrachten.
    piece_position_ids = position_ids[:, 1:]

    for b in range(batch_size):
        # Finde die Indizes, die keine Padding-Tokens sind
        # position_ids für Padding sind 0. piece_position_ids sind niemals 0, außer bei padding.
        valid_indices = torch.where(piece_position_ids[b] > 0)[0]

        for i, piece_idx in enumerate(valid_indices):
            if piece_idx >= local_policies.shape[1]:
                continue

            square = piece_position_ids[b, piece_idx] - POSITION_START_IDX
            if square < 0 or square >= 64:
                continue

            global_start_idx = int(square * 73)
            global_policy[b, global_start_idx : global_start_idx + 73] = local_policies[b, piece_idx, :]

    return global_policy


class AlphaZeroLoss(nn.Module):
    """
    Kombinierte Loss-Funktion für das AlphaZero-Modell.
    Verwendet MSE für den Value und Cross-Entropy für die Policy.
    """
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # CrossEntropyLoss kombiniert log_softmax und NLLLoss. Es erwartet rohe Logits.
        self.policy_loss_fn = nn.CrossEntropyLoss()

    def forward(self, predicted_policy_logits, true_policy_probs, predicted_value, true_value):
        # 1. Value Loss
        value_loss = self.mse_loss(predicted_value.squeeze(-1), true_value)

        # 2. Policy Loss
        # CrossEntropyLoss erwartet (Batch, C) und (Batch,).
        # Unsere `true_policy_probs` sind (Batch, C), also ist es ein Soft-Label-Szenario.
        # Die Standard-CrossEntropyLoss unterstützt standardmäßig keine Soft-Labels.
        # Die manuelle Berechnung ist hier tatsächlich der korrekte Weg für Soft-Labels.
        # loss = sum_i(p_i * log(q_i)), wobei p die Zielwahrscheinlichkeiten und q die vorhergesagten sind.
        policy_log_softmax = torch.log_softmax(predicted_policy_logits, dim=1)
        policy_loss = -torch.sum(true_policy_probs * policy_log_softmax) / predicted_policy_logits.size(0)

        # 3. Kombinierter Loss
        total_loss = value_loss + policy_loss
        return total_loss, value_loss, policy_loss

class ChessDataset(Dataset):
    """Benutzerdefiniertes Dataset für Schach-Trainingsdaten."""
    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        token_ids, position_ids, policy, value = self.training_data[idx]
        return token_ids, position_ids, torch.tensor(policy, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)

def collate_fn(batch):
    """
    Verarbeitet einen Batch von Sequenzen unterschiedlicher Länge durch Padding.
    """
    token_ids, position_ids, policies, values = zip(*batch)

    token_ids_padded = pad_sequence(token_ids, batch_first=True, padding_value=0)
    position_ids_padded = pad_sequence(position_ids, batch_first=True, padding_value=0)
    padding_mask = (token_ids_padded == 0)

    policies = torch.stack(policies)
    values = torch.stack(values)

    return token_ids_padded, position_ids_padded, padding_mask, policies, values

def train_model(model: ChessTransformer, training_data: list, epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001):
    """
    Trainiert das Transformer-Netzwerk.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = AlphaZeroLoss()

    dataset = ChessDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(epochs):
        total_epoch_loss, total_v_loss, total_p_loss = 0, 0, 0
        for batch_tokens, batch_positions, batch_mask, batch_policies, batch_values in dataloader:
            batch_tokens, batch_positions, batch_mask, batch_policies, batch_values = (
                t.to(device) for t in [batch_tokens, batch_positions, batch_mask, batch_policies, batch_values]
            )

            optimizer.zero_grad()

            pred_local_policies, pred_values = model(batch_tokens, batch_positions, src_key_padding_mask=batch_mask)
            pred_global_policies = map_local_to_global_policy(pred_local_policies, batch_positions, device=device)
            loss, v_loss, p_loss = loss_fn(pred_global_policies, batch_policies, pred_values, batch_values)

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        avg_v_loss = total_v_loss / len(dataloader)
        avg_p_loss = total_p_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg Value Loss: {avg_v_loss:.4f}, Avg Policy Loss: {avg_p_loss:.4f}")