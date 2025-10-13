import torch
from neural_network import ChessModel
from main_loop import main_loop
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
    # Sie sind so gewählt, dass ein Zyklus auf einer T4-GPU in Colab
    # in einem vernünftigen Zeitrahmen abgeschlossen werden kann.
    main_loop(
        num_iterations=1,                # Nur eine vollständige Iteration
        num_games_per_iteration=20,      # Mehr Partien für eine bessere Datenbasis
        num_simulations_per_move=100,    # Eine moderate Anzahl an Simulationen für qualitativ gute Züge
        num_eval_games=10,               # Genügend Evaluationspartien für ein aussagekräftiges Ergebnis
        epochs_per_training=5,           # Genügend Epochen, um aus den neuen Daten zu lernen
        acceptance_threshold=0.55
    )

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