# Adversarial text detector

## Dependencies

* ktrain (install by : pip3 install ktrain)

## Usage
### Generating adversarial texts

* Copy 4 output files from adversarial_text_generator:
- `adv.txt`
- `org.txt`
- `adv_predict.txt`
- `org_predict.txt`
	
### Training classifier
* Download `aclImdb.zip` from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA), decompress and place the folder `aclImdb` in the root folder.
* Run `training.py` or use command like `python3 training.py --model bert-large-cased --learning_rate 1e-5 --batch_size 4 --epochs 2` to train a classifier. You can train other models which are defined in here (https://huggingface.co/transformers/pretrained_models.html).

### Predicting classifier
* Run `predicting.py`

##  Acknowledgments

- Code refer to: ktrain ([Github](https://github.com/amaiya/ktrain)).

