from .gpt2 import \
    EnglishGptSmall, EnglishGptMedium, EnglishGptLarge, EnglishGptXl, \
    EnglishPythia70M, EnglishPythia160M, EnglishPythia410M, \
    EnglishPythia14B, EnglishPythia28B, EnglishPythia69B, EnglishPythia120B

MODELS = {
    "gpt-small": EnglishGptSmall,
    "gpt-medium": EnglishGptMedium,
    "gpt-large": EnglishGptLarge,
    "gpt-xl": EnglishGptXl,
    "pythia-70m": EnglishPythia70M,
    "pythia-160m": EnglishPythia160M,
    "pythia-410m": EnglishPythia410M,
    "pythia-14b": EnglishPythia14B,
    "pythia-28b": EnglishPythia28B,
    "pythia-69b": EnglishPythia69B,
    "pythia-120b": EnglishPythia120B,
}


def get_model(model_name):
    model_cls = MODELS[model_name]
    return model_cls()

def get_bow_symbol(model_name):
    return MODELS[model_name].bow_symbol
