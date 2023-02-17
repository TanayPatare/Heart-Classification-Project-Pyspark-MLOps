from flask import Flask, render_template, request
from Classification.utils.util import load_dataset_for_prediction,read_yaml_file,write_yaml_file
import os
import joblib
from Classification.constant import *
MODEL_DIR = os.path.join(ROOT_DIR, MODEL_CHECK)
from Classification.pipeline.pipeline import Pipeline
from Classification.config.configuration import Configuration
from Classification.entity.Heart_classifier import HeartClassifier


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/math', methods=['POST'])  # This will be called from UI
def math_operation():
    MODEL_DIR = os.path.join(ROOT_DIR, MODEL_CHECK)

    if (request.method == 'POST'):
        variable1 = int(request.form['age'])
        variable2 = int(request.form['sex'])
        variable3 = int(request.form['cp'])
        variable4 = int(request.form['trestbps'])
        variable5 = int(request.form['chol'])
        variable6 = int(request.form['fbs'])
        variable7 = int(request.form['restecg'])
        variable8 = int(request.form['thalach'])
        variable9 = int(request.form['exang'])
        variable10 = int(request.form['oldpeak'])
        variable11 = int(request.form['slope'])
        variable12 = int(request.form['ca'])
        variable13 = int(request.form['thal'])

    L =[]
    L.append(int(variable1))
    L.append(int(variable6))
    L.append(int(variable7))
    L.append(int(variable2))
    L.append(int(variable3))   
    L.append(int(variable8))
    L.append(int(variable9))
    L.append(int(variable4))
    L.append(int(variable10))
    L.append(int(variable5))
    L.append(int(variable11))
    L.append(int(variable12))
    L.append(int(variable13))

    L1 = [L]

    column = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    pred_df = load_dataset_for_prediction(L1,column)

    if os.path.exists(MODEL_DIR):
        best_model_config = read_yaml_file(file_path=MODEL_DIR)
        best_model_dir = best_model_config[BEST_MODEL_KEY][MODEL_PATH_KEY]
    else:
        train()
        best_model_config = read_yaml_file(file_path=MODEL_DIR)
        best_model_dir = best_model_config[BEST_MODEL_KEY][MODEL_PATH_KEY]

    x = HeartClassifier(best_model_dir)
    y = x.predict(pred_df)
    
    if y == 1:
        result = "Your heart is at very high risk"
    else:
        result = "Your heart is at no risk"
    results = result
    return render_template('results.html', result=results)


@app.route('/train', methods=['GET', 'POST'])
def train():

    data = {
        BEST_MODEL_KEY: {
                MODEL_PATH_KEY: "",
                MODEL_TRAINER_INDICATOR: 0
                }
            }
    eval_file_path_for_check =  os.path.join(ROOT_DIR, MODEL_CHECK)
    if os.path.exists(path = eval_file_path_for_check): 
        model_eval_content = read_yaml_file(file_path=eval_file_path_for_check)
        if model_eval_content[BEST_MODEL_KEY][MODEL_TRAINER_INDICATOR] == 0:
            data.update({MODEL_TRAINER_INDICATOR:1})
            write_yaml_file(file_path=eval_file_path_for_check,data=data)
            pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
            pipeline.run_pipeline()
            data= read_yaml_file(file_path=eval_file_path_for_check)
            data.update({MODEL_TRAINER_INDICATOR:0})
            write_yaml_file(file_path=eval_file_path_for_check,data=data)
        else:
            pass
    else:
        write_yaml_file(file_path=eval_file_path_for_check,data=data)
        model_eval_content = read_yaml_file(file_path=eval_file_path_for_check)
        if model_eval_content[BEST_MODEL_KEY][MODEL_TRAINER_INDICATOR] == 0:
            data.update({MODEL_TRAINER_INDICATOR:1})
            write_yaml_file(file_path=eval_file_path_for_check,data=data)
            pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
            pipeline.run_pipeline()
            data= read_yaml_file(file_path=eval_file_path_for_check)
            data.update({MODEL_TRAINER_INDICATOR:0})
            write_yaml_file(file_path=eval_file_path_for_check,data=data)
        else:
            pass
    
    return render_template('retrain_ui.html')
    
if __name__ == "__main__":
    app.run()