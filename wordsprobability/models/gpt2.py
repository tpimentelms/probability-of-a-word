import string
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPTNeoXForCausalLM, AutoTokenizer

from wordsprobability.utils.constants import STRIDE


class GPTBaseModel(ABC):
    bow_symbol = 'Ġ'
    model_name = None

    def __init__(self):
        self.model, self.tokenizer = self.get_model()

        self.device = self.model.device
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.get_vocab_masks()

        self.metrics = {
            'surprisal': self._get_surprisal,
            'subword': self._get_tokens,
            'bos_fix': self._get_bos_fix,
            'bow_fix': self._get_bow_fix,
        }

    def get_vocab_masks(self):
        vocab = self.tokenizer.get_vocab()
        n_logits = self.model.embed_out.out_features

        bow_subwords = set(idx for word, idx in vocab.items()
                           if word[0] == self.bow_symbol)
        self.bow_mask = torch.zeros(n_logits)
        self.bow_mask[torch.LongTensor(list(bow_subwords))] = 1

        start_of_punctuation = set(idx for word, idx in vocab.items()
                                   if word[0] in string.punctuation)
        self.punct_mask = torch.zeros(n_logits)
        self.punct_mask[torch.LongTensor(list(start_of_punctuation))] = 1
        self.punct_mask[self.tokenizer.eos_token_id] = 0

        self.eos_mask = torch.zeros(n_logits)
        self.eos_mask[self.tokenizer.eos_token_id] = 1

        self.useless_mask = torch.zeros(n_logits)
        self.useless_mask[self.tokenizer.vocab_size:] = 1

        self.midword_mask = \
            torch.ones(n_logits) - self.bow_mask - self.punct_mask - self.useless_mask
        assert ((self.midword_mask == 1) | (self.midword_mask ==0)).all()

    def get_model(self):
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        return model, tokenizer

    def get_predictions(self, sentence, use_bos_symbol=True):
        return self.get_models_output(sentence, use_bos_symbol=use_bos_symbol)

    def get_models_output(self, sentence, use_bos_symbol=True):
        with torch.no_grad():
            all_results = {metric: torch.tensor([], device=self.device)
                           for metric in self.metrics}
            offset_mapping = []
            start_ind = 0

            while True:
                encodings = self.tokenizer(sentence[start_ind:], max_length=1022,
                                           truncation=True, return_offsets_mapping=True)
                if use_bos_symbol:
                    tensor_input = torch.tensor(
                        [[self.bos_token_id] + encodings['input_ids'] + [self.eos_token_id]],
                        device=self.device)
                else:
                    tensor_input = torch.tensor(
                        [encodings['input_ids'] + [self.eos_token_id]], device=self.device)
                output = self.model(tensor_input, labels=tensor_input)
                shift_logits = output['logits'][..., :-1, :].contiguous()
                shift_labels = tensor_input[..., 1:].contiguous()

                offset = 0 if start_ind == 0 else STRIDE - 1

                # Get metrics needed for analysis
                for metric, metric_fn in self.metrics.items():
                    results = metric_fn(shift_logits, shift_labels, output, tensor_input)
                    all_results[metric] = np.concatenate([all_results[metric], results[offset:-1]])

                offset_mapping.extend([(i + start_ind, j + start_ind)
                                       for i, j in encodings['offset_mapping'][offset:]])
                if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                    break
                start_ind += encodings['offset_mapping'][-STRIDE][1]

            all_results['eow_fix'] = np.concatenate([all_results['bow_fix'][1:], results[-1:]])
            return all_results, offset_mapping

    @staticmethod
    def _get_surprisal(logits, labels, output, _):
        surprisals = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        assert torch.isclose(sum(surprisals) / len(surprisals), output['loss'])

        return surprisals.cpu().numpy()

    def _get_bow_fix(self, logits, _, __, ___):
        probs = F.softmax(logits, dim=-1)
        # ToDo: Should punctuation be a bow as well?
        # bow_fix = - np.log(((self.bow_mask + self.punct_mask) * probs).sum(-1))
        bow_fix = - torch.log(((self.bow_mask + self.eos_mask) * probs).sum(-1))

        return bow_fix.view(-1).cpu().numpy()

    def _get_bos_fix(self, logits, _, __, ___):
        probs = F.softmax(logits, dim=-1)
        bos_fix = - torch.log(
            ((self.midword_mask + self.punct_mask + self.eos_mask) * probs).sum(-1))

        return bos_fix.view(-1).cpu().numpy()

    def _get_tokens(self, _, __, ___, tensor_input):
        return np.array(self.tokenizer.convert_ids_to_tokens(tensor_input[0]))[1:]


class EnglishGptXl(GPTBaseModel):
    language = 'english'
    model_name = 'gpt2-xl'


class EnglishGptLarge(GPTBaseModel):
    language = 'english'
    model_name = 'gpt2-large'


class EnglishGptMedium(GPTBaseModel):
    language = 'english'
    model_name = 'gpt2-medium'


class EnglishGptSmall(GPTBaseModel):
    language = 'english'
    model_name = 'gpt2'


class EnglishPythiaSmall(GPTBaseModel):
    language = 'english'

    def get_model(self):
        model = GPTNeoXForCausalLM.from_pretrained(self.model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer


class EnglishPythia70M(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-70m'

class EnglishPythia160M(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-160m'

class EnglishPythia410M(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-410m'

class EnglishPythia14B(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-1.4b'

class EnglishPythia28B(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-2.8b'

class EnglishPythia69B(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-6.9b'

class EnglishPythia120B(EnglishPythiaSmall):
    model_name = 'EleutherAI/pythia-12b'
