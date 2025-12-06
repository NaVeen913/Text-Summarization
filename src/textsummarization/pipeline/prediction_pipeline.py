from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class PredictionPipeline:
    def __init__(self, model_dir: str = "artifacts/model_trainer/pegasus-text-summarizer"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)

    def predict(self, text: str, max_len: int = 128) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=5,
            max_length=max_len,
            early_stopping=True,
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def predict_batch(self, texts, max_len: int = 128):
        outputs = []
        for t in texts:
            outputs.append(self.predict(t, max_len=max_len))
        return outputs
