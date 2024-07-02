# probability-of-a-word
Code to compute a word's probability using the fixes from "How to Compute the Probability of a Word"


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

The output will be a tsv file with a word per row with its respective computed `surprisal` and `surprisal_fixed` values.
Currently, supported models are: `pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-14b`, `pythia-28b`, `pythia-69b`, `pythia-120b`, `gpt-small`, `gpt-medium`, `gpt-large`, `gpt-xl`.

## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:


```bibtex
@article{pimentel-etal-2024-howto,
    title = "How to Compute the Probability of a Word",
    author = "Pimentel, Tiago and
    Meister, Clara",
    year = "2024",
    eprint = {2406.14561},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2406.14561},
    journal = "arXiv preprint arXiv:2406.14561",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/probability-of-a-word/issues).
