import chess
import math
import numpy as np
import torch
from neural_network import ChessModel
from state_encoder import board_to_tensor

class Node:
    """
    Repräsentiert einen einzelnen Knoten im Monte-Carlo-Suchbaum.
    Jeder Knoten speichert Informationen über einen bestimmten Spielzustand.
    """
    def __init__(self, parent=None, state: chess.Board = None, prior_p: float = 0.0):
        """
        Initialisiert einen Knoten.

        Args:
            parent (Node, optional): Der Elternknoten. Defaults to None (für den Wurzelknoten).
            state (chess.Board): Der Schachbrett-Zustand, den dieser Knoten repräsentiert.
            prior_p (float): Die a-priori-Wahrscheinlichkeit, diesen Knoten auszuwählen,
                             ermittelt vom Policy-Netz.
        """
        self.parent = parent
        self.children: dict[chess.Move, Node] = {} # Mapping von Zug zu Kindknoten
        self.state = state

        self.visit_count = 0
        self.total_value = 0.0 # Summe der Bewertungen aus den Simulationen
        self.prior_probability = prior_p

    def q_value(self) -> float:
        """
        Berechnet den durchschnittlichen Wert (Q-Wert) des Knotens.
        Dies ist der Durchschnitt der Bewertungen aller Simulationen, die durch diesen Knoten liefen.
        Gibt 0 zurück, wenn der Knoten noch nie besucht wurde.

        Returns:
            float: Der Q-Wert des Knotens.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_param: float = 1.41) -> float:
        """
        Berechnet den UCB1-Score für diesen Knoten.
        Diese Formel balanciert zwischen Ausnutzung (hoher Q-Wert) und
        Erkundung (niedrige Besuchszahl).

        Args:
            c_param (float): Der Explorationsparameter.

        Returns:
            float: Der UCB1-Score des Knotens.
        """
        if self.parent is None:
            # Der Wurzelknoten hat keinen UCB-Score im üblichen Sinne
            return 0.0

        exploit_term = self.q_value()
        explore_term = c_param * self.prior_probability * \
                       (math.sqrt(self.parent.visit_count) / (1 + self.visit_count))

        return exploit_term + explore_term

    def select_best_child(self, c_param: float = 1.41) -> 'Node':
        """
        Wählt das beste Kind basierend auf dem UCB1-Score aus.

        Args:
            c_param (float): Der Explorationsparameter.

        Returns:
            Node: Der Kindknoten mit dem höchsten UCB1-Wert.
        """
        return max(self.children.values(), key=lambda child: child.ucb_score(c_param))

    def is_leaf_node(self) -> bool:
        """
        Prüft, ob der Knoten ein Blattknoten ist (d.h. noch nicht expandiert wurde).

        Returns:
            bool: True, wenn der Knoten keine Kinder hat, sonst False.
        """
        return len(self.children) == 0

    def expand(self, policy: dict[chess.Move, float]):
        """
        Expandiert den aktuellen Knoten, indem für jeden legalen Zug ein Kindknoten
        erstellt wird.

        Args:
            policy (dict[chess.Move, float]): Ein Dictionary, das legale Züge auf ihre
                                              a-priori-Wahrscheinlichkeiten vom Policy-Netz abbildet.
        """
        for move, prob in policy.items():
            if move not in self.children:
                next_state = self.state.copy()
                next_state.push(move)
                self.children[move] = Node(parent=self, state=next_state, prior_p=prob)

    def backpropagate(self, value: float):
        """
        Propagiert den Wert einer Simulation vom aktuellen Knoten bis zur Wurzel zurück.
        Aktualisiert `visit_count` und `total_value` für jeden Knoten auf dem Pfad.

        Args:
            value (float): Der Wert aus der Simulation (vom Value-Netz).
        """
        current_node = self
        while current_node is not None:
            current_node.visit_count += 1
            current_node.total_value += value
            value *= -1 # Wert für den nächsten Elternknoten umkehren
            current_node = current_node.parent


from move_translator import move_to_index

def _get_policy_dict(policy_logits: torch.Tensor, board: chess.Board) -> dict[chess.Move, float]:
    """
    Dekodiert die Policy-Logits vom neuronalen Netz in ein Dictionary, das
    legale Züge auf ihre jeweiligen Wahrscheinlichkeiten abbildet.
    """
    # 1. Wende Softmax an, um Logits in Wahrscheinlichkeiten umzuwandeln
    probabilities = torch.softmax(policy_logits, dim=1).squeeze(0)

    policy = {}
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return {}

    for move in legal_moves:
        # 2. Finde den Index für jeden legalen Zug
        index = move_to_index(move, board)
        # 3. Weise die entsprechende Wahrscheinlichkeit zu
        policy[move] = probabilities[index].item()

    # 4. Normalisiere die Wahrscheinlichkeiten der legalen Züge, damit sie zu 1 summieren
    total_prob = sum(policy.values())
    if total_prob > 0:
        for move in policy:
            policy[move] /= total_prob

    return policy


class MCTS:
    """
    Implementiert den Monte-Carlo Tree Search Algorithmus, der von einem
    neuronalen Netz geleitet wird.
    """
    def __init__(self, model: ChessModel, c_param: float = 1.41):
        self.model = model
        self.c_param = c_param
        self.model.eval() # Modell in den Evaluationsmodus schalten

    def _run_simulation(self, root: Node):
        """
        Führt eine einzelne, vollständige MCTS-Simulation aus:
        Selection -> Expansion -> Evaluation -> Backpropagation.
        """
        # 1. Selection: Finde einen Blattknoten
        current_node = root
        while not current_node.is_leaf_node():
            current_node = current_node.select_best_child(self.c_param)

        # 2. Expansion & Evaluation
        if current_node.state.is_game_over():
            # Wenn das Spiel am Blattknoten beendet ist, bestimme den Wert aus dem Ergebnis
            outcome = current_node.state.outcome()
            if outcome.winner is None: # Unentschieden
                value = 0.0
            else: # Matt
                # Der Wert ist aus der Perspektive des Spielers, der am Zug ist.
                # Wenn das Spiel vorbei ist, hat dieser Spieler verloren.
                value = -1.0
        else:
            # Konvertiere den Zustand in einen Tensor für das NN
            state_tensor = board_to_tensor(current_node.state)
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0)

            # Erhalte Policy und Value vom NN
            with torch.no_grad():
                policy_logits, value_tensor = self.model(state_tensor)

            value = value_tensor.item()

            # Expandiere den Knoten mit der Policy vom NN
            policy = _get_policy_dict(policy_logits, current_node.state)
            current_node.expand(policy)

        # 3. Backpropagation
        current_node.backpropagate(value)

    def find_best_move(self, board: chess.Board, num_simulations: int) -> chess.Move:
        """
        Führt die MCTS-Suche für eine gegebene Brettstellung aus und gibt den besten Zug zurück.
        """
        root = Node(state=board.copy())

        # Führe die geplante Anzahl von Simulationen aus
        for _ in range(num_simulations):
            self._run_simulation(root)

        # Wähle den Zug, der zum meistbesuchten Kindknoten führt
        if not root.children:
            return None

        return max(root.children.keys(), key=lambda move: root.children[move].visit_count)