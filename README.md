# LLM CPU vs GPU Performance Tests

This project contains simple, hands-on experiments to compare the inference performance of a Hugging Face transformer model running on **CPU vs GPU**.

Using a sentiment analysis model (`distilbert-base-uncased-finetuned-sst-2-english`), the tests measure how fast the model processes input text in different scenarios:

- **Single inference**: Run the model once and compare the time on CPU and GPU.
- **Batch inference**: Run the model multiple times (e.g., 100 inferences) to see how GPU acceleration scales with workload.
- **CPU advantage scenario**: Demonstrate cases where CPU can be faster than GPU due to lower data transfer and warm-up overhead.

This practical experiment helps understand the trade-offs between CPU and GPU for AI workloads, illustrating when and why GPUs significantly speed up deep learning tasks.
