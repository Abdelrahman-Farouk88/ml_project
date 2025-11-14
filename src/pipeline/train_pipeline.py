import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline...")
            
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion complete. Train path: {train_data_path}, Test path: {test_data_path}")

            # Step 2: Data Transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path, test_path=test_data_path
            )
            logging.info("Data transformation complete.")

            # Step 3: Model Training
            r2_score, best_model_name = self.model_trainer.iniate_model_trainer(
                train_array=train_arr, test_array=test_arr
            )
            logging.info(f"Model training complete. Best model is {best_model_name} with R2 score: {r2_score}")
            
            logging.info("Training pipeline finished successfully.")

        except Exception as e:
            logging.error("Exception occurred in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()