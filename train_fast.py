import torch
import torch.multiprocessing as mp
from transformer_network import ChessTransformer
from self_play_parallel import run_parallel_games
from train import train_model
from evaluate import run_tournament
import os
import time

def train_fast_on_colab():
    """
    Führt einen optimierten Trainingszyklus aus, der darauf ausgelegt ist,
    in ca. 10 Minuten auf einer Google Colab GPU sinnvolle Ergebnisse zu erzielen.

    Diese Funktion dient dazu, schnell ein "best_model.pth" zu erzeugen,
    gegen das man dann mit play_vs_ai.py spielen kann.
    """
    print("Starte optimierten Trainingslauf für eine schnelle Ausführung (~10 Minuten)...")
    start_time = time.time()

    # Diese Hyperparameter sind ein Kompromiss zwischen Geschwindigkeit und Qualität.
    num_games = 20
    num_simulations = 100
    num_eval_games = 10
    epochs = 5

    # Lade oder erstelle das beste Modell
    best_model_path = "models/best_model.pth"
    os.makedirs("models/archive", exist_ok=True)
    if os.path.exists(best_model_path):
        best_model = ChessTransformer()
        best_model.load_state_dict(torch.load(best_model_path))
    else:
        best_model = ChessTransformer()

    # 1. Paralleles Selbstspiel
    training_data = run_parallel_games(best_model, num_games=num_games, num_simulations=num_simulations)

    # 2. Training
    print(f"\nTrainiere neues Modell mit {len(training_data)} Positionen...")
    new_model = ChessTransformer()
    new_model.load_state_dict(best_model.state_dict())
    train_model(new_model, training_data, epochs=epochs)

    # 3. Evaluation
    print(f"\nEvaluiere neues Modell gegen bestes Modell...")
    win_rate = run_tournament(new_model, best_model, num_matches=num_eval_games, num_simulations=num_simulations)

    print(f"\nErgebnis: Gewinnrate des neuen Modells: {win_rate*100:.1f}%")
    if win_rate > 0.55:
        print("Neues Modell wird als 'best_model.pth' gespeichert.")
        torch.save(new_model.state_dict(), best_model_path)
    else:
        print("Bestes Modell wurde nicht ersetzt.")

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print(f"\nSchneller Trainingslauf abgeschlossen in {duration_minutes:.2f} Minuten.")
    print("Ein 'best_model.pth' sollte nun im 'models'-Verzeichnis verfügbar sein.")
    print("Sie können jetzt 'python play_vs_ai.py' ausführen, um dagegen zu spielen.")

if __name__ == "__main__":
    # Überprüfe, ob eine GPU verfügbar ist
    if not torch.cuda.is_available():
        print("WARNUNG: Es wurde keine GPU gefunden. Das Training wird sehr langsam sein.")
        print("Bitte stellen Sie sicher, dass Sie in Ihrer Colab-Umgebung eine GPU-Laufzeit ausgewählt haben.")

    train_fast_on_colab()