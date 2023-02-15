from Classification.logger import logging
from Classification.exception import ClassificationException
from Classification.entity.config_entity import ModelEvaluationConfig
from Classification.entity.artifact_entity import ModelTrainerArtifact,ModelEvaluationArtifact
from Classification.constant import *
import os
import sys
from Classification.utils.util import write_yaml_file, read_yaml_file, load_object,load_dataset
from Classification.entity.model_factory import ModelFactory

class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise ClassificationException(e, sys) from e

    def get_best_model_production(self):
        try:
            # to get the model under production
            # case1: first time execution
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model

            #case2: Not the first run
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            #Not the first run but due to low accuracy model was not created in first run
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise ClassificationException(e, sys) from e

    def update_evaluation_report(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_trainer_artifact.trained_model_file_path,
                    MODEL_ACCURACY: model_trainer_artifact.model_accuracy
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model,}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise ClassificationException(e, sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            newly_trained_model_accuracy = self.model_trainer_artifact.model_accuracy
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)

            if model_eval_content is not None:
                if BEST_MODEL_KEY in model_eval_content:
                    previous_best_model_accuracy = int(model_eval_content[BEST_MODEL_KEY][MODEL_ACCURACY])
                    if newly_trained_model_accuracy > previous_best_model_accuracy:
                        self.update_evaluation_report(self.model_trainer_artifact)
                        model_eval_artifact = ModelEvaluationArtifact(
                            is_model_accepted = True,
                            evaluated_model_path = self.model_trainer_artifact.trained_model_file_path
                        )
                        logging.info(f"Model evaluation completed. model metric artifact")
                        return model_eval_artifact
                    else:
                        previous_best_model = model_eval_content[BEST_MODEL_KEY][MODEL_PATH_KEY]
                        model_eval_artifact = ModelEvaluationArtifact(
                            is_model_accepted = False,
                            evaluated_model_path = previous_best_model
                        )
                        return model_eval_artifact
                else:
                    self.update_evaluation_report(self.model_trainer_artifact)
                    model_eval_artifact = ModelEvaluationArtifact(
                            is_model_accepted = True,
                            evaluated_model_path = self.model_trainer_artifact.trained_model_file_path
                        )
                    return model_eval_artifact
            else:
                self.update_evaluation_report(self.model_trainer_artifact)
                model_eval_artifact = ModelEvaluationArtifact(
                            is_model_accepted = True,
                            evaluated_model_path = self.model_trainer_artifact.trained_model_file_path
                        )
                return model_eval_artifact
        except Exception as e:
            raise ClassificationException(e, sys) from e
    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")