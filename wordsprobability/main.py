import argparse
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

from .models import get_model, get_bow_symbol
from .utils import constants, utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data files
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    # Model
    parser.add_argument('--model', type=str, required=True, choices=constants.MODELS)
    # Extra Options
    parser.add_argument('--return-buggy-surprisals', action='store_true')

    return parser.parse_args()


def get_surprisals_per_subword(text, model_name):
    dfs = []

    model = get_model(model_name)
    for text_id, utterance in tqdm(enumerate(text), total=len(text),
                                   desc='Getting Surprisals'):
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


def agg_surprisal_per_word(df: pd.DataFrame, model_name: str,
                           return_buggy_surprisals: Optional[bool] = False) -> pd.DataFrame:
    bow_symbol = get_bow_symbol(model_name)

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

    df_per_word['word_bytes'] = df_per_word.subword.apply(
        lambda x: x[1:] if (x[0] == bow_symbol) else x)
    df_per_word['word'] = df_per_word.subword_clean.apply(
        lambda x: x.strip())

    return_columns = ['word', 'surprisal', 'surprisal_buggy'] \
        if return_buggy_surprisals else ['word', 'surprisal']
    return df_per_word[return_columns]


def _get_surprisal_per_word(text_list: List[str], model_name: str,
                            return_buggy_surprisals: Optional[bool] = False) -> pd.DataFrame:
    df = get_surprisals_per_subword(text_list, model_name)
    return agg_surprisal_per_word(df, model_name, return_buggy_surprisals)


def get_surprisal_per_word(text: str, model_name: str,
                           return_buggy_surprisals: Optional[bool] = False) -> pd.DataFrame:
    text_list = text.split('\n')
    return _get_surprisal_per_word(text_list, model_name, return_buggy_surprisals)


def main():
    args = get_args()

    text = utils.read_txt(args.input)
    df = _get_surprisal_per_word(text, args.model,
                                 return_buggy_surprisals=args.return_buggy_surprisals)
    utils.write_tsv(df, args.output)


if __name__ == '__main__':
    main()
