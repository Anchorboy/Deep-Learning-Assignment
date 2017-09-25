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
    img_embed_size = 1028
    word_embed_size = 256
    hidden_size = 256
    batch_size = 64
    beam_size = 5
    n_epochs = 1000
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


class Hypothesis:
    def __init__(self, toks, log_prob, state):
        self.toks = toks
        self.log_prob = log_prob
        self.state = state

    def extend(self, tok, log_prob, new_state):
        return Hypothesis(self.toks + [tok], self.log_prob + log_prob, new_state)

    @property
    def latest_token(self):
        return self.toks[-1]

    def __str__(self):
        return ("Hypothesis(log prob = %.4f, tokens = %s)" % (self.log_prob, self.toks))