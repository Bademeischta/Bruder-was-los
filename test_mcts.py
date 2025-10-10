import unittest
import chess
import torch
from mcts import Node, MCTS
from neural_network import ChessModel

class MockChessModel(ChessModel):
    """
    Eine Mock-Version des ChessModel für Testzwecke.
    Gibt eine vordefinierte, gleichmäßige Policy und einen festen Value zurück.
    """
    def forward(self, x):
        # Policy: 4672-dimensionaler Vektor, hier als Platzhalter
        policy_logits = torch.randn(1, 4672)
        # Value: Fester Wert von 0.5
        value = torch.tensor([[0.5]])
        return policy_logits, value

class TestMCTS(unittest.TestCase):
    """Testet die MCTS-Klasse und ihre Komponenten."""

    def setUp(self):
        """Initialisiert eine saubere Umgebung für jeden Test."""
        self.board = chess.Board()
        self.model = MockChessModel()
        self.mcts = MCTS(self.model)

    def test_node_creation(self):
        """Testet die korrekte Initialisierung eines Knotens."""
        root = Node(state=self.board)
        self.assertEqual(root.visit_count, 0)
        self.assertEqual(root.total_value, 0.0)
        self.assertEqual(root.q_value(), 0.0)
        self.assertTrue(root.is_leaf_node())

    def test_expansion(self):
        """Testet, ob ein Knoten korrekt expandiert wird."""
        root = Node(state=self.board)
        legal_moves = list(self.board.legal_moves)
        # Erstelle eine Dummy-Policy
        policy = {move: 1.0 / len(legal_moves) for move in legal_moves}

        root.expand(policy)

        self.assertFalse(root.is_leaf_node())
        self.assertEqual(len(root.children), len(legal_moves))
        # Prüfe, ob ein zufälliger Kindknoten korrekt initialisiert wurde
        sample_move = legal_moves[0]
        child_node = root.children[sample_move]
        self.assertEqual(child_node.parent, root)
        self.assertEqual(child_node.prior_probability, policy[sample_move])

    def test_backpropagation(self):
        """Testet die korrekte Rückpropagierung von Werten."""
        root = Node(state=self.board)
        root.expand({chess.Move.from_uci("e2e4"): 0.5})
        child = root.children[chess.Move.from_uci("e2e4")]

        value = 0.8
        child.backpropagate(value)

        # Kindknoten
        self.assertEqual(child.visit_count, 1)
        self.assertEqual(child.total_value, 0.8)

        # Wurzelknoten
        self.assertEqual(root.visit_count, 1)
        # Der Wert wird für den Elternknoten negiert
        self.assertEqual(root.total_value, -0.8)

    def test_mcts_single_simulation(self):
        """Testet eine einzelne MCTS-Simulation auf dem Wurzelknoten."""
        root = Node(state=self.board.copy())
        self.mcts._run_simulation(root)

        # Nach einer Simulation:
        # 1. Der Wurzelknoten wurde besucht.
        self.assertEqual(root.visit_count, 1)
        # 2. Der Wert vom Mock-Modell (0.5) wurde zurückpropagiert.
        #    Der Wert wird für den Elternknoten (hier None) negiert, also bleibt er für den Knoten selbst positiv.
        self.assertEqual(root.total_value, 0.5)
        # 3. Der Wurzelknoten wurde expandiert.
        self.assertFalse(root.is_leaf_node())
        self.assertEqual(len(root.children), len(list(self.board.legal_moves)))

    def test_find_best_move(self):
        """Testet die Hauptfunktion zur Zugfindung."""
        # Führe eine kleine Anzahl von Simulationen aus
        best_move = self.mcts.find_best_move(self.board, num_simulations=10)

        # Das Ergebnis sollte ein valider Zug sein
        self.assertIsInstance(best_move, chess.Move)
        self.assertIn(best_move, self.board.legal_moves)

if __name__ == '__main__':
    unittest.main()