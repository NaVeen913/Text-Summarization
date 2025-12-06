from textsummarization.config.configuration import ConfigurationManager
from textsummarization.components.data_ingestion import DataIngestion
from textsummarization.components.data_transformation import DataTransformation
from textsummarization.components.model_trainer import ModelTrainer
from textsummarization.logging import logger


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run_data_ingestion(self):
        data_ingestion_config = self.config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.run()

    def run_data_transformation(self):
        data_transformation_config = self.config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.run()

    def run_model_trainer(self):
        model_trainer_config = self.config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.run()

    def run(self):
        logger.info("======== PIPELINE STARTED ========")
        self.run_data_ingestion()
        self.run_data_transformation()
        self.run_model_trainer()
        logger.info("======== PIPELINE FINISHED ========")





