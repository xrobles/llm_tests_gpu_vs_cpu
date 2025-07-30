import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Load tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample input text
text = "NVIDIA GPUs are amazing for AI acceleration!"
inputs = tokenizer(text, return_tensors="pt")

# ======================
# RUN ON CPU
# ======================
model_cpu = model.to("cpu")
inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}

# --- Single Inference on CPU ---
start = time.time()
with torch.no_grad():
    outputs_cpu = model_cpu(**inputs_cpu)
end = time.time()
print(f"✅ CPU - Single Inference Time: {end - start:.4f} seconds")

# --- 100 Inferences on CPU ---
start = time.time()
with torch.no_grad():
    for _ in range(100):
        outputs_cpu = model_cpu(**inputs_cpu)
end = time.time()
print(f"✅ CPU - 100 Inference Time: {end - start:.4f} seconds")

# ======================
# RUN ON GPU
# ======================
if torch.cuda.is_available():
    model_gpu = model.to("cuda")
    inputs_gpu = {k: v.to("cuda") for k, v in inputs.items()}

    # Warm-up GPU
    with torch.no_grad():
        _ = model_gpu(**inputs_gpu)

    # --- Single Inference on GPU ---
    start = time.time()
    with torch.no_grad():
        outputs_gpu = model_gpu(**inputs_gpu)
    end = time.time()
    print(f"🚀 GPU - Single Inference Time: {end - start:.4f} seconds")

    # --- 100 Inferences on GPU ---
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs_gpu = model_gpu(**inputs_gpu)
    end = time.time()
    print(f"🚀 GPU - 100 Inference Time: {end - start:.4f} seconds")

else:
    print("⚠️ GPU is not available in this environment.")
