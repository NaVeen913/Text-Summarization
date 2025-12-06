import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from textsummarization.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    pipe = TrainingPipeline()

    print("\n=== STEP 1: Data Ingestion ===")
    pipe.run_data_ingestion()

    print("\n=== STEP 2: Data Transformation ===")
    pipe.run_data_transformation()

    print("\n=== STEP 3: Model Training ===")
    pipe.run_model_trainer()

    print("\n ALL STEPS COMPLETED")









