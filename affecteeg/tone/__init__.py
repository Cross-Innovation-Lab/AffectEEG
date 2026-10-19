"""Main-experiment tone decoding and AffectEEG adaptation."""

def run(args):
    from .training import run as train
    return train(args)


def evaluate(args):
    from .evaluation import evaluate as evaluate_selected
    return evaluate_selected(args)
