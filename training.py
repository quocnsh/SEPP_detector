import os
import numpy as np
import ktrain
from ktrain import text
import re
import argparse

parser = argparse.ArgumentParser(description='Train model with imdb dataset.')
parser.add_argument("-m", "--model", 
                    help="Model name",                     
                    default = "bert-large-cased")
parser.add_argument("-lr", "--learning_rate", 
                    help="Learning rate",                     
                    type=float, default = 1e-5)
parser.add_argument("-b", "--batch_size", 
                    help="Batch size",                     
                    type=int, default = 2)
parser.add_argument("-e", "--epochs", 
                    help="Epochs",                     
                    type=int, default = 2)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

def rm_tags(text):
    """
    Remove tags
    Input: 
        text: input text
    Output:
        re_tag: text after removing tags
    """
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

def read_imdb_files(filetype):
    """
    Read imdb files from 'train' or 'test'
    Input: 
        filetype: type of set : 'train' or 'test'    
    Output:
        all_texts: list of texts
        all_labels: list of labels
    """
    all_labels = []
    for _ in range(12500):
        all_labels.append([0, 1])
    for _ in range(12500):
        all_labels.append([1, 0])

    all_texts = []
    file_list = []
    path = r'./aclImdb/'
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    return all_texts, all_labels

def split_imdb_files():
    """
    split imdb files into 'train' and 'test'
    Output:
        train_texts: list of train texts
        train_labels: list of train labels
        test_texts: list of test texts
        test_labels: list of test labels       
    """
    print('Processing IMDB dataset')
    train_texts, train_labels = read_imdb_files('train')
    test_texts, test_labels = read_imdb_files('test')
    return train_texts, train_labels, test_texts, test_labels

def train_model(model_name, batch_size, learning_rate, epochs):
    """
    train and save a model
    Input:
        model_name: name of model from Hunggingface (https://huggingface.co/transformers/pretrained_models.html)
        batch_size: batch size of the model
        learning_rate: learning rate for training
        epochs: number of epochs for training
    """    
    x_train, y_train, x_test, y_test = split_imdb_files();
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    target_names = ["pos","neg"]
    t = text.Transformer(model_name, maxlen=512, class_names=target_names)
    trn = t.preprocess_train(x_train, y_train)
    val = t.preprocess_test(x_test, y_test)
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size)
    learner.fit_onecycle(learning_rate, epochs)
    learner.validate(class_names=t.get_classes())
    learner.view_top_losses(n=1, preproc=t)
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    
    if not os.path.exists("./imdb_model"):
        os.mkdir("./imdb_model")
    predictor.save(f'./imdb_model/{model_name}')
    
if __name__ == "__main__":
    train_model(args.model, args.batch_size, args.learning_rate, args.epochs)