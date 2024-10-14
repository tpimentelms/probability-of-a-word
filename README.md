# probability-of-a-word

[![CircleCI](https://circleci.com/gh/tpimentelms/probability-of-a-word.svg?style=svg)](https://circleci.com/gh/tpimentelms/probability-of-a-word)

Code to compute a word's probability using the fixes from "How to Compute the Probability of a Word"

For the code to reproduce our paper's experiments, see [this repository](https://github.com/tpimentelms/probability-of-a-word-experiments).


### Installation

You can install WordsProbability directly from PyPI:

`pip install wordsprobability`

Or from source:

```
git clone git@github.com:tpimentelms/probability-of-a-word.git
cd probability-of-a-word
pip install -e .
```

#### Dependencies

WordsProbability has the following requirements:

* [Pandas](https://pandas.pydata.org)
* [PyTorch](https://pytorch.org)
* [Transformers](https://huggingface.co/docs/transformers/en/index)

### Usage

#### Basic Usage

Install this repository. Then run:
```bash
$ wordsprobability --model pythia-70m --input examples/abstract.txt --output temp.tsv
```

The input must be a txt file, with one sequence per line.
The output will be a tsv file with a word per row with its respective computed `surprisal` values.
To also get computed `surprisal_buggy` values (without our paper's correction) use the optional flag `--return-buggy-surprisals`.
Currently, supported models are: `pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-14b`, `pythia-28b`, `pythia-69b`, `pythia-120b`, `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`.

#### Using in other Applications

Import wordsprobability in your application and get surprisals with:
```python
    from wordsprobability import get_surprisal_per_word
    df = get_surprisal_per_word(text='Hello world! Who are you???\nWho am I?', model_name='pythia-70m')
```


## Extra Information

#### Citation

If this code or the paper were useful to you, consider citing it:


```bibtex
@inproceedings{pimentel-etal-2024-howto,
    title = "How to Compute the Probability of a Word",
    author = "Pimentel, Tiago and
    Meister, Clara",
    year = "2024",
    url = {https://arxiv.org/abs/2406.14561},
    booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
    publisher = {Association for Computational Linguistics},
    address = {Miami, Florida, USA},
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/probability-of-a-word/issues).
