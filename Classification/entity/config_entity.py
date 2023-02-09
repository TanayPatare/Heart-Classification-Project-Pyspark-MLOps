from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
["dataset_download_url","rar_download_dir","raw_data_dir","ingested_train_dirr","ingested_test_dirr"])

"""
1. dataset_download_url : it consists of dataset url
2. csv_download_dir : it will consist of folder name for dataset
3. ingested_train_dirr : it will consist of folder name for train dataset
4. ingested_test_dirr : it will consist of folder name for test dataset
"""

DataValidationConfig = namedtuple("DataValidationConfig",
["schema_file_path"])

DataTransformationConfig = ("DataTransformationConfig",
["transform_train_dir","transform_test_dir","preprocessed_object_file_path"])

"""
1. transform_train_dir : it will consist of folder name for transformed train dataset
2. transform_test_dir : it will consist of folder name for transformed train dataset
3. preprocessed_object_file_path : it will consist of folder name for preprocessing pickle file
"""

ModelTrainerConfig = namedtuple("ModelTrainerConfig",
["trained_model_file_path","base_accuracy"])

"""
1. trained_model_file_path : it will contain the file path of the new model created
2. base_accuracy : it is the base accuracy which model should achieve 
"""

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig",
["model_evaluation_file_path","time_stamp"])

"""
1. model_evaluation_file_path : it will contain the file path of model under production
"""

ModelPusherConfig = namedtuple("ModelPusherConfig",
["export_dir_path"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",
["artifact_dir"])