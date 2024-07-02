def write_tsv(df, fname):
    df.to_csv(fname, sep='\t')


def read_txt(fname):
    text = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            text += [line]
    return text
