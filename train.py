import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformer_network import ChessTransformer
from self_play import play_game # Für die Generierung von Dummy-Daten

class AlphaZeroLoss(nn.Module):
    """
    Kombinierte Loss-Funktion für das AlphaZero-Modell.
    Besteht aus dem Mean Squared Error für den Value und dem Cross-Entropy-Loss
    für die Policy.
    """
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_policy, true_policy, predicted_value, true_value, policy_active=True):
        # Value Loss
        value_loss = self.mse_loss(predicted_value.squeeze(-1), true_value)

        if not policy_active:
            return value_loss

        # Policy Loss (muss noch für die lokale Policy des Transformers angepasst werden)
        # Placeholder:
        policy_loss = 0.0
        # policy_loss = -torch.sum(true_policy * torch.log_softmax(predicted_policy, dim=1)) / predicted_policy.size()[0]

        total_loss = value_loss + policy_loss
        return total_loss

class ChessDataset(Dataset):
    """Benutzerdefiniertes Dataset für Schach-Trainingsdaten."""
    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        # Unpacke die tokenisierten Daten
        token_ids, position_ids, policy, value = self.training_data[idx]
        return token_ids, position_ids, torch.tensor(policy, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)

def collate_fn(batch):
    """
    Verarbeitet einen Batch von Sequenzen unterschiedlicher Länge, indem es
    Padding hinzufügt, um sie auf die gleiche Länge zu bringen.
    """
    token_ids, position_ids, policies, values = zip(*batch)

    # Padding für Token- und Positions-IDs
    token_ids_padded = pad_sequence(token_ids, batch_first=True, padding_value=0)
    position_ids_padded = pad_sequence(position_ids, batch_first=True, padding_value=0)

    # Erstelle eine Padding-Maske
    # True, wo die Tokens Padding sind, False sonst
    padding_mask = (token_ids_padded == 0)

    policies = torch.stack(policies)
    values = torch.stack(values)

    return token_ids_padded, position_ids_padded, padding_mask, policies, values

def train_model(model: ChessTransformer, training_data: list, epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001):
    """
    Trainiert das Transformer-Netzwerk mit den gesammelten Trainingsdaten.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = AlphaZeroLoss()

    dataset = ChessDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        for batch_tokens, batch_positions, batch_mask, batch_policies, batch_values in dataloader:
            optimizer.zero_grad()

            # Forward-Pass
            pred_local_policies, pred_values = model(batch_tokens, batch_positions, src_key_padding_mask=batch_mask)

            # TODO: Der Loss muss angepasst werden, um die lokale Policy zu verarbeiten.
            # Dies ist eine komplexe Änderung. Für den Moment verwenden wir einen vereinfachten Loss.
            # Wir berechnen den Loss nur auf dem Value-Head.
            loss = loss_fn(None, batch_policies, pred_values, batch_values, policy_active=False)

            # Backward-Pass und Optimierung
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Durchschnittlicher Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    print("Starte Beispiel für Trainings-Modul...")
    # 1. Erzeuge Dummy-Modell und Dummy-Daten
    dummy_model = ChessModel()
    print("Generiere Dummy-Trainingsdaten aus 1 Partie...")
    # Generiere Daten aus einer kurzen Partie
    dummy_training_data = play_game(dummy_model, num_simulations=8, exploration_moves=5)

    # 2. Trainiere das Modell mit den Dummy-Daten
    print(f"\nBeginne Training mit {len(dummy_training_data)} Datensätzen...")
    train_model(dummy_model, dummy_training_data, epochs=5, batch_size=16)

    print("\nTrainings-Modul-Beispiel erfolgreich abgeschlossen.")