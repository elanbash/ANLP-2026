# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on the MRPC dataset.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py \
  --do_train \
  --do_predict \
  --num_train_epochs <number_of_training_epochs> \
  --lr <learning_rate> \
  --batch_size <batch_size> \
  --model_path <path_to_save_or_load_model> \
  --max_train_samples <optional_limit_or_-1> \
  --max_eval_samples <optional_limit_or_-1> \
  --max_predict_samples <optional_limit_or_-1>
```

If you use --do_train the model will fine-tune on the training data and save its weights to the specified --model_path, and if you use --do_predict, a predictions.txt file will be generated in that same path containing prediction results for all test samples.