## Language Model
Dataset
- MLDS 2017

#### Requirement
- Python 2.7.10
- tensorflow 1.2.0

### Use
- `config.py - Configuration`
- `preprocess.py - Preprocessing`
- `data_loader.py - Input pipeline`
- `s2s_model.py - Nets architecture`
- `main.py - Main entry point`

Use preprocessing before running.

#### example
1. `config.py`
Calling for configuration.
`config = Config(args)`

2. `preprocess.py`
Preprocessing function for both **training/** **testing/** data.
- Training/ Dev
`def preprocess(source_dir, target_dir)`

- Testing
`def preprocess_test(target_dir)`

- Output
```
train.json/ dev.json/ test.json
{
  "data": array(shape=(n_samples, max_length), dtype=int),
  "n_samples": int
}
param.plk - vocab_table, parameters:(max_length, vocab_size)
```

3. `data_loader.py`
Control dataflow & preprocessing for labels/ mask.
```
def get_batch(batch_size):
  return (inputs_batch, labels_batch, mask_batch)
```

4. `s2s_model.py`
Provide 3 types of cell
- RNN
- GRU
- LSTM
Prediction is determined by MIN loss in a group of 5 examples.

5. `main.py`
- `def do_test - testing training RNN model`
- `def do_train - training RNN model`
- `def do_evaluate - predict the word in the ___ and pick the answer from a,b,c,d,e`
