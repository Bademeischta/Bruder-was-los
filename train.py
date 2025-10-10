import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from neural_network import ChessModel
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
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predicted_policy, true_policy, predicted_value, true_value):
        # Value Loss
        value_loss = self.mse_loss(predicted_value.squeeze(), true_value)

        # Policy Loss
        policy_loss = -torch.sum(true_policy * torch.log_softmax(predicted_policy, dim=1)) / predicted_policy.size()[0]

        total_loss = value_loss + policy_loss
        return total_loss

def train_model(model: ChessModel, training_data: list, epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001):
    """
    Trainiert das neuronale Netzwerk mit den gesammelten Trainingsdaten.

    Args:
        model (ChessModel): Das zu trainierende Modell.
        training_data (list): Eine Liste von Trainings-Tupeln.
        epochs (int): Die Anzahl der Trainings-Epochen.
        batch_size (int): Die Größe der Batches für das Training.
        learning_rate (float): Die Lernrate für den Optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # L2 Regularisierung
    loss_fn = AlphaZeroLoss()

    # Konvertiere die Trainingsdaten in PyTorch-Tensoren
    states, policies, values = zip(*training_data)
    states_tensor = torch.from_numpy(np.array(states))
    policies_tensor = torch.from_numpy(np.array(policies))
    values_tensor = torch.from_numpy(np.array(values, dtype=np.float32))

    dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train() # Modell in den Trainingsmodus schalten
    for epoch in range(epochs):
        total_epoch_loss = 0
        for batch_states, batch_policies, batch_values in dataloader:
            optimizer.zero_grad()

            # Forward-Pass
            pred_policies, pred_values = model(batch_states)

            # Loss berechnen
            loss = loss_fn(pred_policies, batch_policies, pred_values, batch_values)

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