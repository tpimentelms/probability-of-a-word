import argparse
import pandas as pd

from .models import get_model, get_bow_symbol
from .utils import constants, utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data files
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    # Model
    parser.add_argument('--model', type=str, required=True, choices=constants.MODELS)

    return parser.parse_args()


def get_surprisals_per_subword(args):
    text = utils.read_txt(args.input)
    dfs = []

    model = get_model(args.model)
    for text_id, utterance in enumerate(text):
        results, offsets = model.get_predictions(utterance.strip())

        df = pd.DataFrame(results)
        df['text_id'] = text_id
        df['offsets'] = offsets
        dfs += [df]

    return pd.concat(dfs)


def mark_bos_subwords(df):
    df['ones'] = 1
    df['text_pos'] = df.groupby('text_id')['ones'].cumsum()
    df['is_bos'] = df['text_pos'] == 1
    del df['ones']
    del df['text_pos']

    return df


def mark_eow_subwords(df):
    df['is_eow'] = df.groupby('text_id')['is_bow'].shift(-1)
    df.loc[df['is_eow'].isna(), 'is_eow'] = True

    return df


def agg_surprisal_per_word(df, args):
    bow_symbol = get_bow_symbol(args.model)

    df['is_bow'] = df.subword.apply(lambda x: x[0] == bow_symbol)
    df = mark_bos_subwords(df)
    df = mark_eow_subwords(df)
    df['word_id'] = df.groupby('text_id')['is_bow'].cumsum()

    df['surprisal_buggy'] = df['surprisal']
    df['surprisal'] = df['surprisal'] \
                      - df['bow_fix'] * df['is_bow'] \
                      - df['bos_fix'] * df['is_bos'] \
                      + df['eow_fix'] * df['is_eow']

    df_per_word = df.groupby(['text_id', 'word_id']).agg('sum')

    assert ((df_per_word.is_bow + df_per_word.is_bos) == 1).all()
    assert (df_per_word.is_eow == 1).all()

    df_per_word['word'] = df_per_word.subword.apply(
        lambda x: x[1:] if (x[0] == bow_symbol) else x)

    return df_per_word[['word', 'surprisal', 'surprisal_buggy']]


def main():
    args = get_args()
    df = get_surprisals_per_subword(args)
    df = agg_surprisal_per_word(df, args)
    utils.write_tsv(df, args.output)


if __name__ == '__main__':
    main()
