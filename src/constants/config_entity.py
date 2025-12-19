from src.constants.entity import DataIngestionEntity, ArtifactEntity
from src.constants.entity import DataTransformationEntity, ModelTrainingEntity
from src.constants.entity import PredictionEntity


class ArtifactConfig:
    def __init__(self):
        artifact_entity = ArtifactEntity()
        self.artifact_name = artifact_entity.ARTIFACT_DIR_NAME


class DataIngestionConfig:
    def __init__(self):
        # Data Ingestion Entity Object
        data_ingestion_entity = DataIngestionEntity()
        # paths for the data ingestion class
        self.data_ingestion_config = data_ingestion_entity.DATA_INGESTION_DIR_PATH
        self.data_csv_config = data_ingestion_entity.NEO_DATA_FILE_NAME
        self.data_table_name = data_ingestion_entity.NEAR_EARTH_OBJECT_TABLE


class DataValidationConfig:
    def __init__(self):
        pass


class DataTransformationConfig:
    def __init__(self):
        # Data Transformation Entity Object
        data_transformtion_entity = DataTransformationEntity()

        # paths for transformation object
        self.preprocessing_table_name = data_transformtion_entity.PREPROCESSING_PICKLE_TABLE
        self.label_encoder_table_name = data_transformtion_entity.LABEL_ENCODER_PICKLE_TABLE
        self.features = data_transformtion_entity.FEATURE_COLUMNS
        self.target = data_transformtion_entity.TARGET_COLUMNS
        self.drop_columns = data_transformtion_entity.DROP_COLUMNS

        # num and cat cols
        self.cat_cols = data_transformtion_entity.CATEGORICAL_COLUMNS
        self.num_cols = data_transformtion_entity.NUMERICAL_COLUMNS


class ModelTrainerConfig:
    def __init__(self):
        model_training_entity = ModelTrainingEntity()
        self.model_params = model_training_entity.model_params
        self.parameter_scoring = model_training_entity.hyper_parameter_scores


class PredictionConfig:
    def __init__(self):
        prediction_entity = PredictionEntity()
        self.prediction_log_table = prediction_entity.predict_table_name
