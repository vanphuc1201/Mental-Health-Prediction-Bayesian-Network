import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("features/bn_features_raw_test.csv")
df.columns = df.columns.str.strip()
texts = df["text"].tolist()
labels = df["stress_label"].tolist()

# =========================
# 2. Setup Pipeline (Optimized for Mac/MPS)
# =========================
# Note: Use "meta-llama/Llama-3.2-3B-Instruct" for better prompt following
model_id = "Qwen/Qwen2.5-3B-Instruct"

# Use device="mps" if on Apple Silicon, or "cuda" for NVIDIA
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=device
)

# =========================
# 3. Prediction logic
# =========================
predictions = []

for i, text in enumerate(texts):
    # Llama-3-Instruct works best with this specific format
    messages = [
        {"role": "system", "content": "You are a classifier. Respond ONLY with '0' or '1'."},
        {"role": "user", "content": f"Classify stress (1=stressed, 0=not): {text}"},
    ]
    
    # Generate
    outputs = pipe(messages, max_new_tokens=2, temperature=0.1)
    ans = outputs[0]["generated_text"][-1]['content'].strip()
    
    # Parse output safely
    pred = 1 if "1" in ans else 0
    predictions.append(pred)
    
    if (i + 1) % 5 == 0:
        print(f"Processed {i+1}/{len(texts)}")

# =========================
# 4. Detailed Evaluation
# =========================
print("\n=== Performance Metrics ===")
print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
print(f"F1 Score: {f1_score(labels, predictions):.4f}")
print("\nDetailed Report:\n", classification_report(labels, predictions))