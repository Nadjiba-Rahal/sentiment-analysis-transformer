import subprocess
import sys

def run_training():
    print("Starting training...")
    subprocess.run([sys.executable, "train.py", "config.yaml"])

def run_evaluation():
    print("Running evaluation...")
    subprocess.run([sys.executable, "visualize.py"])

def test_model():
    print("Testing model prediction...")
    subprocess.run([
        sys.executable,
        "predict.py",
        "This movie was absolutely phenomenal!"
    ])

if __name__ == "__main__":
    print("\n=== SENTIMENT ANALYSIS PIPELINE ===\n")

    run_training()
    run_evaluation()
    test_model()

    print("\n Completed successfully!")