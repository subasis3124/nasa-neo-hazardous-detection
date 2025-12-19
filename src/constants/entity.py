from src.constants.params import PARAMS, SCORING


""" ----DEFINING ALL THE TABLE NAMES USED -----"""


class TableNameEntity:
    def __init__(self):
        self.NEAR_EARTH_OBJECT_TABLE: str = 'train_neo'


""" ----------ARTIFACT DIRECTORY ----------"""


class ArtifactEntity:
    def __init__(self):
        self.ARTIFACT_DIR_NAME: str = 'artifacts'


""" --------DATA INGESTION ARTIFACT ---------"""


class DataIngestionEntity(TableNameEntity):
    def __init__(self):
        super().__init__()
        self.DATA_INGESTION_DIR_PATH: str = 'Data'
        self.NEO_DATA_FILE_NAME: str = 'neo_data.csv'


"""-----DATA VALIDATION ARTIFACT ENTITY-----"""


"""------DATA TRANSFORMATION ARTIFACT ENTITY------"""


class DataTransformationEntity(ArtifactEntity):
    def __init__(self):
        super().__init__()
        self.DATA_TRANSFORMATION_DIR_PATH: str = 'data_transformation'
        self.PREPROCESSING_PICKLE_TABLE: str = 'preprocessing_table'
        self.LABEL_ENCODER_PICKLE_TABLE:  str = 'label_encoder_table'
        self.DROP_COLUMNS: list = ['id', 'name', 'close_approach_date', 'close_approach_date_full', 'nasa_jpl_url', 'orbiting_body', 'max_diameter_km', 'min_diameter_km']
        self.FEATURE_COLUMNS: list = ['absolute_magnitude_h', 'epoch_date_close_approach', 'relative_velocity_kps', 'miss_distance_km', 'is_sentry_object', 'diameter_range']
        self.TARGET_COLUMNS: list = ['is_potentially_hazardous']

        # PREPROCESSIN NUMERICAL AND CATEGORICAL COLUMNS
        self.CATEGORICAL_COLUMNS: list = ['is_sentry_object']
        self.NUMERICAL_COLUMNS: list = ['absolute_magnitude_h', 'epoch_date_close_approach', 'relative_velocity_kps', 'miss_distance_km', 'diameter_range']


"""---------MODEL TRAINING ARTIFACT ENTITY---------"""


class ModelTrainingEntity:
    def __init__(self):
        self.model_params: dict = PARAMS
        self.hyper_parameter_scores: dict = SCORING


"""-----PREDICTING ARTIFACT ENTITY-----"""


class PredictionEntity:
    def __init__(self):
        self.predict_table_name: str = 'prediction_table'


class AsteroidRiskAnalysisEntity:
    def __init__(self):
        self.RISK_ANALYSIS_TABLE_NAME: str = 'risk_analysis'
        self.COLS_SHOWS: list = ['id', 'davg_m', 'mass_kg', 'impact_energy_j', 'Enorm', 'Rmoid_norm', 'Phazardous', 'RiskScore_raw', 'RiskScorenorm', 'RiskCategory']
        self.MANUAL_WEIGHTS: list = [0.4, 0.4, 0.2]
