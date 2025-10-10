import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Ein residueller Block, wie er in der AlphaZero-Architektur verwendet wird.
    Besteht aus zwei Convolutional Layern mit einer Skip-Connection.
    """
    def __init__(self, num_channels=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Skip-Connection
        out = F.relu(out)
        return out

class ChessModel(nn.Module):
    """
    Die vollständige neuronale Netzwerkarchitektur, angelehnt an AlphaZero.
    Besteht aus einem gemeinsamen Body und zwei spezialisierten Heads (Policy und Value).
    """
    def __init__(self, num_residual_blocks=19, num_input_channels=21, num_policy_outputs=4672):
        super(ChessModel, self).__init__()

        # 1. Gemeinsamer Body
        self.input_conv = nn.Sequential(
            nn.Conv2d(num_input_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        # 2. Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, num_policy_outputs)
        )

        # 3. Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Skaliert den Output auf [-1, 1]
        )

    def forward(self, x):
        """
        Führt den Forward-Pass durch das Netzwerk aus.

        Args:
            x (torch.Tensor): Der Input-Tensor mit der Dimension (N, 21, 8, 8).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Ein Tupel bestehend aus:
                - policy_logits (N, 4672): Die Logits für die Zugwahrscheinlichkeiten.
                - value (N, 1): Die Bewertung der Position.
        """
        # Durch den Body
        x = self.input_conv(x)
        x = self.residual_blocks(x)

        # Durch die Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

if __name__ == '__main__':
    # Beispiel für die Instanziierung und einen Forward-Pass
    # Dies dient der Verifikation der Dimensionen und der grundlegenden Funktionalität
    model = ChessModel()

    # Erzeuge einen zufälligen Input-Tensor, der einem Batch von 4 Brettstellungen entspricht
    dummy_input = torch.randn(4, 21, 8, 8)

    # Führe den Forward-Pass aus
    policy_logits, value = model(dummy_input)

    print("--- Modell-Architektur ---")
    print(model)
    print("\n--- Dimensions-Check ---")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Policy Shape: {policy_logits.shape}") # Erwartet: (4, 4672)
    print(f"Value Shape:  {value.shape}")      # Erwartet: (4, 1)

    assert policy_logits.shape == (4, 4672)
    assert value.shape == (4, 1)
    print("\nDimensions-Check erfolgreich!")