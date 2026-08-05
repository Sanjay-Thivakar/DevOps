from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.evaluation import ModelEvaluation


if __name__ == "__main__":

    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    X_train, X_test, y_train, y_test, _ = (
        transformation.initiate_data_transformation()
    )

    trainer = ModelTrainer()

    model = trainer.initiate_model_training(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluator = ModelEvaluation()

    metrics = evaluator.initiate_model_evaluation(
        model,
        X_test,
        y_test
    )

    print(metrics)