from typing import Dict, Any, Tuple, Callable, Iterator, List, Optional, Iterable

import numpy as np
import torch

from classy.data.data_drivers import QASample
from classy.data.dataset.base import batchify
from classy.data.dataset.hf.classification import HFQADataset


import logging

from classy.utils.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class ESCEDDataset(HFQADataset):
    def __init__(
        self,
        samples_iterator: Callable[[], Iterator[QASample]],
        vocabulary: Vocabulary,
        transformer_model: str,
        additional_special_tokens: Optional[List[str]],
        tokens_per_batch: int,
        max_batch_size: Optional[int],
        section_size: int,
        prebatch: bool,
        materialize: bool,
        min_length: int,
        max_length: int,
        for_inference: bool,
        candidates_separator: Optional[str] = None,
        shuffle_candidates_prob: float = 0.0,
    ):
        self.candidates_separator = candidates_separator
        self.shuffle_candidates_prob = shuffle_candidates_prob

        super().__init__(
            samples_iterator=samples_iterator,
            vocabulary=vocabulary,
            transformer_model=transformer_model,
            additional_special_tokens=additional_special_tokens,
            tokens_per_batch=tokens_per_batch,
            max_batch_size=max_batch_size,
            section_size=section_size,
            prebatch=prebatch,
            materialize=materialize,
            min_length=min_length,
            max_length=max_length,
            for_inference=for_inference,
        )

    def _compute_char_to_tokens(
        self, qa_sample: QASample, elem_dict: Dict[str, Any]
    ) -> Dict[int, int]:
        # use token2chars to build the mapping char2token
        # we should be using tokenization_output.char_to_token but there
        # seems to be some bug around it with some tokenizers (e.g. BartTokenizer)
        # t("Question", "X Y").char_to_token(1, sequence_id=1) returns None
        # (first paper to char_to_token is 1 to account for added prefix space)
        char2token = {}
        first = True
        for _t_idx, (m, cp) in enumerate(
            zip(elem_dict["context_mask"].tolist(), elem_dict["token2chars"].tolist())
        ):
            if m:

                # postprocess token2chars
                # some tokenizers (microsoft/deberta-base) include in the token-offsets also the white space
                # e.g. 'In Italy' => ' Italy' => (2, 8)
                # set position to first non-white space
                while (
                    elem_dict["token2chars"][_t_idx][0]
                    < elem_dict["token2chars"][_t_idx][1]
                    and qa_sample.context[elem_dict["token2chars"][_t_idx][0].item()]
                    == " "
                ):
                    elem_dict["token2chars"][_t_idx][0] += 1

                # add prefix space seems to be bugged on some tokenizers
                if first:
                    first = False
                    if cp[0] != 0 and qa_sample.context[cp[0] - 1] != " ":
                        # this is needed to cope with tokenizers such as bart
                        # where t("Question", "X Y").token2chars[<bpe of X>] = (1, 1)
                        elem_dict["token2chars"][_t_idx][0] -= 1
                        cp = (cp[0] - 1, cp[1])
                if cp[0] == cp[1]:
                    # this happens on some tokenizers when multiple spaces are present
                    assert (
                        qa_sample.context[cp[0] - 1] == " "
                    ), f"Token {_t_idx} found to occur at char span ({cp[0]}, {cp[1]}), which is impossible"
                for c in range(*cp):
                    char2token[c] = _t_idx

        return char2token

    def process_candidates(
        self, qa_sample: QASample, shuffle_candidates: bool
    ) -> Tuple[QASample, Dict[str, Tuple[int, int]]]:
        if shuffle_candidates:
            np.random.shuffle(qa_sample.candidates)

        answer = (
            qa_sample.context[qa_sample.char_start : qa_sample.char_end]
            if qa_sample.char_start is not None
            else None
        )
        context = ""
        candidates_offsets = dict()
        answer_start, answer_end = None, None
        for candidate in qa_sample.candidates:
            candidate_start = len(context)
            candidate_end = candidate_start + len(candidate)
            candidates_offsets[candidate] = candidate_start, candidate_end
            if candidate == answer:
                answer_start, answer_end = candidate_start, candidate_end
            candidate = candidate + f" {self.candidates_separator} "
            context += candidate

        qa_sample.context = context
        qa_sample.char_start = answer_start
        qa_sample.char_end = answer_end

        return qa_sample, candidates_offsets

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for qa_sample in self.samples_iterator():
            qa_sample: QASample
            shuffle_candidates = (
                self.shuffle_candidates_prob > 0
                and np.random.uniform() < self.shuffle_candidates_prob
            )

            qa_sample, candidates_offsets = self.process_candidates(
                qa_sample, shuffle_candidates
            )

            tokenization_output = self.tokenizer(
                qa_sample.question,
                qa_sample.context,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            elem_dict = {
                "input_ids": tokenization_output["input_ids"].squeeze(0),
                "attention_mask": tokenization_output["attention_mask"].squeeze(0),
                "token2chars": tokenization_output["offset_mapping"].squeeze(0),
                "context_mask": torch.tensor(
                    [p == 1 for p in tokenization_output.sequence_ids()]
                ),
            }

            char2token = self._compute_char_to_tokens(qa_sample, elem_dict)

            elem_dict["candidates_offsets"] = {
                candidate: (char2token[si], char2token[ei - 1])
                for candidate, (si, ei) in candidates_offsets.items()
            }

            if "token_type_ids" in tokenization_output:
                elem_dict["token_type_ids"] = tokenization_output[
                    "token_type_ids"
                ].squeeze(0)

            if not self.for_inference:

                if qa_sample.reference_annotation is None:
                    logger.info(
                        "Found an instance with no gold labels while 'for_inference' is set to False. Skipping"
                    )
                    continue

                (
                    reference_char_start,
                    reference_char_end,
                ) = qa_sample.reference_annotation

                elem_dict["start_position"] = char2token[reference_char_start]
                elem_dict["end_position"] = char2token[reference_char_end - 1]

                if (
                    elem_dict["start_position"] is None
                    or elem_dict["end_position"] is None
                ):
                    logger.warning(
                        "Skipping instance since either the start or the end position are None. "
                        "This is probably due to a tokenizer error"
                    )
                    continue

            elem_dict["samples"] = qa_sample

            yield elem_dict

    def init_fields_batcher(self) -> Dict:
        return {
            "input_ids": lambda lst: batchify(
                lst, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "token_type_ids": lambda lst: batchify(lst, padding_value=0),
            "context_mask": lambda lst: batchify(lst, padding_value=0),
            "token2chars": None,
            "start_position": lambda lst: torch.tensor(lst, dtype=torch.long),
            "end_position": lambda lst: torch.tensor(lst, dtype=torch.long),
            "samples": None,
            "candidates_offsets": None,
        }
