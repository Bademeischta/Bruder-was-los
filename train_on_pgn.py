import argparse
import torch
import os
from transformer_network import ChessTransformer
from pgn_parser import parse_pgn_file
from train import train_model

def main():
    """
    Hauptfunktion zum Trainieren eines Modells mit Daten aus einer PGN-Datei.
    """
    parser = argparse.ArgumentParser(
        description="Trainiert das Transformer-Modell mit Partien aus einer PGN-Datei."
    )
    parser.add_argument(
        "pgn_file",
        type=str,
        help="Der Pfad zur PGN-Datei, die für das Training verwendet werden soll."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Die Anzahl der Trainings-Epochen."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Die Batch-Größe für das Training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Die Lernrate für den Optimizer."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/pretrained_model.pth",
        help="Der Pfad, unter dem das trainierte Modell gespeichert wird."
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Optional: Pfad zu einem bestehenden Modell, um das Training fortzusetzen."
    )

    args = parser.parse_args()

    # 1. Parse die PGN-Datei
    print(f"Lese und parse die PGN-Datei: {args.pgn_file}...")
    if not os.path.exists(args.pgn_file):
        print(f"Fehler: Die Datei '{args.pgn_file}' wurde nicht gefunden.")
        return

    training_data = parse_pgn_file(args.pgn_file)
    if not training_data:
        print("Keine Trainingsdaten gefunden in der PGN-Datei.")
        return

    print(f"{len(training_data)} Trainingspositionen extrahiert.")

    # 2. Lade oder erstelle das Modell
    model = ChessTransformer()
    if args.load_path:
        if os.path.exists(args.load_path):
            print(f"Lade bestehendes Modell von: {args.load_path}")
            model.load_state_dict(torch.load(args.load_path))
        else:
            print(f"Warnung: Das angegebene Modell '{args.load_path}' wurde nicht gefunden. Starte mit einem neuen Modell.")
    else:
        print("Erstelle ein neues, zufällig initialisiertes Modell.")

    # 3. Starte das Training
    print(f"\nBeginne Training für {args.epochs} Epochen...")
    train_model(
        model,
        training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # 4. Speichere das trainierte Modell
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nTraining abgeschlossen. Modell gespeichert unter: {args.save_path}")

if __name__ == "__main__":
    main()