import unittest
import torch
from neural_network import ChessModel, ResidualBlock

class TestNeuralNetwork(unittest.TestCase):
    """
    Testet die `ChessModel`-Klasse und ihre Komponenten, um die korrekte
    Architektur und die Dimensionsstimmigkeit sicherzustellen.
    """

    def test_residual_block(self):
        """Testet, ob ein ResidualBlock die Dimensionen des Tensors beibehält."""
        block = ResidualBlock(num_channels=256)
        # Dummy-Input mit Batch-Größe 4
        dummy_input = torch.randn(4, 256, 8, 8)
        output = block(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_chess_model_instantiation(self):
        """Testet, ob das ChessModel-Objekt erfolgreich instanziiert werden kann."""
        try:
            model = ChessModel(num_residual_blocks=19, num_input_channels=21, num_policy_outputs=4672)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"ChessModel-Instanziierung fehlgeschlagen mit Fehler: {e}")

    def test_chess_model_forward_pass_dimensions(self):
        """
        Testet den Forward-Pass des Modells und verifiziert die Dimensionen
        des Policy- und Value-Outputs.
        """
        model = ChessModel(num_residual_blocks=19, num_input_channels=21, num_policy_outputs=4672)
        model.eval() # In den Evaluationsmodus schalten

        # Erzeuge einen zufälligen Input-Tensor, der einem Batch von 4 Brettstellungen entspricht
        batch_size = 4
        dummy_input = torch.randn(batch_size, 21, 8, 8)

        with torch.no_grad(): # Keine Gradientenberechnung für Inferenz
            policy_logits, value = model(dummy_input)

        # Überprüfe die Output-Dimensionen
        self.assertEqual(policy_logits.shape, (batch_size, 4672))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_value_head_output_range(self):
        """Stellt sicher, dass der Output des Value Heads im Bereich [-1, 1] liegt."""
        model = ChessModel()
        model.eval()

        dummy_input = torch.randn(8, 21, 8, 8) # Test mit größerem Batch

        with torch.no_grad():
            _, value = model(dummy_input)

        # Prüft, ob alle Werte im Tensor zwischen -1 und 1 liegen
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))

if __name__ == '__main__':
    unittest.main()