from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from textsummarization.entity.config_entity import ModelTrainerConfig
from textsummarization.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def run(self) -> None:
        logger.info(">>> Running Model Trainer...")

        # 1. Load datasets
        logger.info("Loading tokenized datasets from disk...")
        train_ds = load_from_disk(str(self.config.train_dataset_path))
        val_ds = load_from_disk(str(self.config.val_dataset_path))

        # 2. Load model & tokenizer
        logger.info(f"Loading model and tokenizer: {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # 3. Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.output_dir),
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            predict_with_generate=self.config.predict_with_generate,
        )

        # 4. Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # 5. Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished.")

        # 6. Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        logger.info("Model saved.")

