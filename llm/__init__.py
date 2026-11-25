"""Fine-tuned LLM recommender (Qwen2.5 + LoRA), trained on the cross-domain user histories.

- llm.data       user split, prompts and the training / test examples
- llm.train      LoRA fine-tuning
- llm.recommend  LLMRecommender: generate titles, match them to the real catalogue
- llm.evaluate   fair comparison with the co-occurrence engine and popularity

Needs the extra packages in requirements-llm.txt (and a CUDA GPU for training).
"""

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = "models/llamarec-qwen2.5-1.5b-lora"
