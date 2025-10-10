import chess

class ChessCore:
    """
    Diese Klasse kapselt die Kernlogik eines Schachspiels und stellt eine
    präzise Schnittstelle zur Steuerung und Abfrage des Spielzustands bereit.
    Sie verwendet die python-chess Bibliothek für die interne Brett-Repräsentation.
    """
    def __init__(self):
        """
        Initialisiert eine neue Schachpartie.
        Das Schachbrett (`chess.Board`) wird als private Instanzvariable `self.__board` geführt.
        """
        self.__board = chess.Board()

    def get_legal_moves(self) -> list[str]:
        """
        Gibt eine Liste aller legalen Züge im UCI-Format zurück.

        Returns:
            list[str]: Eine Liste von Zügen wie ['g1f3', 'h2h4', ...].
        """
        return [move.uci() for move in self.__board.legal_moves]

    def push_uci(self, move_uci: str) -> bool:
        """
        Führt einen Zug mittels UCI-String aus, nachdem dieser validiert wurde.

        Args:
            move_uci (str): Der auszuführende Zug im UCI-Format (z.B. 'e2e4').

        Returns:
            bool: True, wenn der Zug legal war und ausgeführt wurde, andernfalls False.
                  Der Zustand des Bretts ändert sich nur bei einem legalen Zug.
        """
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.__board.legal_moves:
                self.__board.push(move)
                return True
            return False
        except ValueError:
            # Ungültiges UCI-Format
            return False

    def get_game_status(self) -> str:
        """
        Ermittelt den aktuellen Spielstatus.

        Returns:
            str: "checkmate", "stalemate" oder "in_progress".
                 Andere Unentschieden-Regeln sind hier nicht implementiert.
        """
        # Zuerst auf Matt prüfen, da dies die spezifischste Endbedingung ist.
        if self.__board.is_checkmate():
            return "checkmate"

        # is_game_over() prüft auf alle Endbedingungen (Matt, Patt, Regeln).
        # Da Matt bereits ausgeschlossen wurde, deckt dies alle anderen Fälle ab
        # (Stalemate, Insufficient Material etc.), die als "stalemate" gelten sollen.
        if self.__board.is_game_over():
            return "stalemate"

        # Wenn keine Endbedingung erfüllt ist, läuft das Spiel.
        return "in_progress"

    def get_board_fen(self) -> str:
        """
        Gibt die FEN-Repräsentation des aktuellen Bretts zurück.

        Returns:
            str: Die FEN-Zeichenfolge des Bretts.
        """
        return self.__board.fen()

    def load_fen(self, fen_string: str) -> bool:
        """
        Setzt das interne Board auf einen durch einen FEN-String definierten Zustand.

        Args:
            fen_string (str): Der FEN-String, der geladen werden soll.

        Returns:
            bool: True, wenn der FEN-String valide war und geladen wurde, sonst False.
        """
        try:
            self.__board.set_fen(fen_string)
            return True
        except ValueError:
            # Ungültiger FEN-String
            return False