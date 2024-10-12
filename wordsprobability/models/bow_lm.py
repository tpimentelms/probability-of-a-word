import string
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPTNeoXForCausalLM, AutoTokenizer


class BaseBOWModel(ABC):
    bow_symbol = 'Ġ'
    model_cls = None
    tokenizer_cls = None
    model_name = None

    def __init__(self):
        self.model, self.tokenizer = self._initialise_model()

        self.device = self.model.device
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self._initialise_vocab_masks()

        self.metrics = {
            'surprisal': self._get_surprisal,
            'subword': self._get_tokens,
            'bos_fix': self._get_bos_fix,
            'bow_fix': self._get_bow_fix,
        }

    def _initialise_model(self):
        model = self.model_cls.from_pretrained(self.model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        elif torch.backends.mps.is_available():
            model.to('mps')
        tokenizer = self.tokenizer_cls.from_pretrained(self.model_name)
        return model, tokenizer

    def _initialise_vocab_masks(self):
        vocab = self.tokenizer.get_vocab()
        n_logits = self._get_n_logits()

        self.vocab_masks = {}
        self.vocab_masks['bow'] = self._initialise_bow_mask(vocab, n_logits)
        self.vocab_masks['punct'] = self._initialise_punct_mask(vocab, n_logits)
        self.vocab_masks['eos'] = self._initialise_eos_mask(n_logits)
        self.vocab_masks['useless'] = self._initialise_useless_mask(n_logits)
        self.vocab_masks['mid'] = self._initialise_midword_mask(n_logits, self.vocab_masks)

    @abstractmethod
    def _get_n_logits(self):
        raise NotImplementedError

    def _initialise_bow_mask(self, vocab, n_logits):
        bow_subwords = set(idx for word, idx in vocab.items()
                           if word[0] == self.bow_symbol)
        bow_mask = torch.zeros(n_logits, device=self.device)
        bow_mask[torch.LongTensor(list(bow_subwords))] = 1

        return bow_mask

    def _initialise_punct_mask(self, vocab, n_logits):
        start_of_punctuation = set(idx for word, idx in vocab.items()
                                   if word[0] in string.punctuation)
        punct_mask = torch.zeros(n_logits, device=self.device)
        punct_mask[torch.LongTensor(list(start_of_punctuation))] = 1
        punct_mask[self.tokenizer.eos_token_id] = 0

        return punct_mask

    def _initialise_eos_mask(self, n_logits):
        eos_mask = torch.zeros(n_logits, device=self.device)
        eos_mask[self.tokenizer.eos_token_id] = 1

        return eos_mask

    def _initialise_useless_mask(self, n_logits):
        useless_mask = torch.zeros(n_logits, device=self.device)
        useless_mask[self.tokenizer.vocab_size:] = 1
        return useless_mask

    def _initialise_midword_mask(self, n_logits, vocab_masks):
        midword_mask = torch.ones(n_logits, device=self.device) - vocab_masks['bow'] \
                       - vocab_masks['punct'] - vocab_masks['useless']
        return midword_mask

    def get_predictions(self, sentence, use_bos_symbol=True):
        return self.get_models_output(sentence, use_bos_symbol=use_bos_symbol)

    def get_models_output(self, sentence, use_bos_symbol=True, stride=200):
        with torch.no_grad():
            all_results = {metric: torch.tensor([], device='cpu')
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

                offset = 0 if start_ind == 0 else stride - 1

                # Get metrics needed for analysis
                for metric, metric_fn in self.metrics.items():
                    results = metric_fn(shift_logits, shift_labels, output, tensor_input)
                    all_results[metric] = np.concatenate([all_results[metric], results[offset:-1]])

                offset_mapping.extend([(i + start_ind, j + start_ind)
                                       for i, j in encodings['offset_mapping'][offset:]])
                if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                    break
                start_ind += encodings['offset_mapping'][-stride][1]

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
        bow_vocab = self.vocab_masks['bow'] + self.vocab_masks['eos']
        bow_fix = - torch.log((bow_vocab * probs).sum(-1))

        return bow_fix.view(-1).cpu().numpy()

    def _get_bos_fix(self, logits, _, __, ___):
        probs = F.softmax(logits, dim=-1)
        bos_vocab = self.vocab_masks['mid'] + self.vocab_masks['punct'] + self.vocab_masks['eos']
        bos_fix = - torch.log((bos_vocab * probs).sum(-1))

        return bos_fix.view(-1).cpu().numpy()

    def _get_tokens(self, _, __, ___, tensor_input):
        return np.array(self.tokenizer.convert_ids_to_tokens(tensor_input[0]))[1:]


class GPT2BaseModel(BaseBOWModel, ABC):
    language = 'english'
    model_cls = GPT2LMHeadModel
    tokenizer_cls = GPT2TokenizerFast

    def _get_n_logits(self):
        return self.model.lm_head.out_features


class EnglishGpt2Xl(GPT2BaseModel):
    model_name = 'gpt2-xl'


class EnglishGpt2Large(GPT2BaseModel):
    model_name = 'gpt2-large'


class EnglishGpt2Medium(GPT2BaseModel):
    model_name = 'gpt2-medium'


class EnglishGpt2Small(GPT2BaseModel):
    model_name = 'gpt2'


class PythiaBaseModel(BaseBOWModel, ABC):
    language = 'english'
    model_cls = GPTNeoXForCausalLM
    tokenizer_cls = AutoTokenizer

    def _get_n_logits(self):
        return self.model.embed_out.out_features


class EnglishPythia70M(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-70m'


class EnglishPythia160M(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-160m'


class EnglishPythia410M(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-410m'


class EnglishPythia14B(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-1.4b'


class EnglishPythia28B(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-2.8b'


class EnglishPythia69B(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-6.9b'


class EnglishPythia120B(PythiaBaseModel):
    model_name = 'EleutherAI/pythia-12b'
