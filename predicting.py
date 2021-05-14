import os
import ktrain
from ktrain import text
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

MODEL_FOLDER = "imdb_model"
MODELS = [ f.name for f in os.scandir(MODEL_FOLDER) if f.is_dir() ]
ORG_VICTIM_PREDICT_FILE = r"data/org_predict.txt"
ADV_VICTIM_PREDICT_FILE = r"data/adv_predict.txt"  
ORG_FILE =  r"data/org.txt"
ADV_FILE = r"data/adv.txt"

def read_all_lines(input_file):
    """
    Read all lines in a file
    Input: 
        input_file: path of the file
    Output:
        result: lines in the file
    """
    fi = open(input_file,"r", encoding="utf8") 
    lines = fi.readlines()
    fi.close()
    result = []
    for line in lines:
        result.append(line.strip())
    return result

def get_victim_predict(filename):
    """
    Get predictions from a victim
    Input: 
        filename: path of the file, which contains predictions, seperated by comma
    Output:
        predict: list of predictions
    """
    lines = read_all_lines(filename)
    predict = []
    for line in lines:
        probs = line.split(",")
        predict.append([float(probs[0]), float(probs[1])])
    return predict

def predict_text(input_lines, model_paths):
    """
    Get predictions from all trained models
    Input: 
        input_lines: list of texts
        model_paths: paths of models
    Output:
        predictions: list of predictions for all models
    """
    predictions = []
    for model_path in model_paths:
        print(f"predicting with {model_path}")
        model_folder = os.path.join(f"./{MODEL_FOLDER}/{model_path}")
        reloaded_predictor = ktrain.load_predictor(model_folder)        
        predic_proba = reloaded_predictor.predict(input_lines, return_proba=True)
        predictions.append(predic_proba)
    return predictions

def extract_feature(victim_predict, supplemental_predict):
    """
    Extract feature 
    Input: 
        predict: victim predict in predict[0], others are predictions from other classifiers
    Output:
        feature: feature extraction
    """
    feature = []
    for index, victim in enumerate(victim_predict):
        victim_predicted_class = np.argmax(victim)
        sub_feature = []
        different_count = 0
        for supplemental in supplemental_predict:
            distance = victim[victim_predicted_class]-supplemental[index][victim_predicted_class]
            distance = np.asscalar(distance)
            sub_feature.append(abs(distance))
            if np.argmax(supplemental[index]) != victim_predicted_class:
                different_count += 1
        sub_feature.append(different_count)
        feature.append(sub_feature)
    feature = np.array(feature)
    return feature
    
def get_predict(org_file, adv_file, org_victim_predict_file, adv_victim_predict_file, model_paths):
    """
    Get predict 
    Input: 
        org_file: file path of original text
        adv_file: file path of adversarial text
        org_victim_predict_file: file path of victim prediction for original text
        adv_victim_predict_file: file path of victim prediction for adveresarial text
        model_paths: paths of supplemental classifiers
    Output:
        accuracy: accuracy of the prediction
    """
    org_text = read_all_lines(org_file)
    adv_text = read_all_lines(adv_file)
    org_victim_predict = get_victim_predict(org_victim_predict_file)
    adv_victim_predict = get_victim_predict(adv_victim_predict_file)
    num_sample = len(adv_victim_predict)
    org_victim_predict = np.array(org_victim_predict)
    adv_victim_predict= np.array(adv_victim_predict)
    org_predict = predict_text(org_text, model_paths)
    adv_predict = predict_text(adv_text, model_paths)
    org_feature = extract_feature(org_victim_predict, org_predict)
    adv_feature = extract_feature(adv_victim_predict, adv_predict)
    y_org = np.zeros(num_sample)
    y_adv = np.ones(num_sample)
    x_feature = np.concatenate([org_feature, adv_feature], axis=0)
    y_feature = np.concatenate([y_org, y_adv])
    X_train, X_test, y_train, y_test = train_test_split(x_feature, y_feature, test_size=0.1, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)    
    return accuracy

if __name__ == '__main__':    
    accuracy = get_predict(ORG_FILE, ADV_FILE, ORG_VICTIM_PREDICT_FILE, ADV_VICTIM_PREDICT_FILE, MODELS)
    print(f"accuracy = {accuracy}")  
