from datetime import *

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

	Create test here.
    """
    img_length = 80
    cap_length = 40
    n_img_features = 4096
    max_length = 120 # longest sequence to parse
    vocab_size = 3000
    dropout = 0.667
    embed_size = 500
    hidden_size = 256
    batch_size = 64
    n_epochs = 20
    lr = 0.001
    max_grad_norm = 5.

    def __init__(self, args):
        self.cell = args.cell
        self.mode = args.mode

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"
