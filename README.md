# kaggle_sleep_state
kaggle icr competition https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states

# Set up data

```
kaggle competitions download -c child-mind-institute-detect-sleep-states -p data/original_data

```

Then unzip

# How To

Run
- preprocess.py to create traning dataset
- train.py --train to train a lgbm with classification setup