# LLM CPU vs GPU Inference Time Comparison

This project contains a simple script to compare the inference time of a Hugging Face transformer model running on **CPU vs GPU**.

The script uses the `distilbert-base-uncased-finetuned-sst-2-english` model for sentiment analysis and measures the time for:

- **Single inference** on CPU and GPU
- **100 inferences** on CPU and GPU

It also includes a check to see if a GPU is available and prints a message if not.

This test helps demonstrate:

- The overhead of running a single inference on GPU versus CPU
- The performance gain of running multiple inferences on GPU compared to CPU

Use this experiment to better understand when GPU acceleration helps in NLP model inference tasks.
