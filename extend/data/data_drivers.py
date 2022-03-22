from typing import Iterator, Generator, List, Optional, Tuple, Dict

import hydra.utils
from classy.data.data_drivers import QASample, QADataDriver, READERS_DICT, QA

import re
import json
import html
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)


def build_context(
    candidates: List[str], answer: Optional[str]
) -> Tuple[str, Optional[int], Optional[int]]:
    context = ""
    answer_start, answer_end = None, None

    for candidate in candidates:

        if answer is not None and candidate == answer:
            answer_start = len(context)
            answer_end = answer_start + len(candidate)

        candidate = candidate + " . "
        context += candidate

    return context, answer_start, answer_end


class AidaDataDriver(QADataDriver):
    def read(self, lines: Iterator[str]) -> Generator[QASample, None, None]:

        impossible_answers = 0

        for line in lines:

            json_obj = json.loads(line.strip())
            meta_info = json_obj["meta"]

            # building question
            left_context = meta_info["left_context"]
            mention = meta_info["mention"]
            right_context = meta_info["right_context"]
            question = " ".join([left_context, "{", mention, "}", right_context])

            # building context
            candidates = json_obj["candidates"]

            if len(candidates) == 0:
                candidates = ["FAKE-CANDIDATE"]

            answer = json_obj["output"][0]["answer"]

            context, answer_start, answer_end = build_context(candidates, answer)

            if answer_start is None or answer_end is None:
                impossible_answers += 1

            yield QASample(
                html.unescape(context),
                html.unescape(question),
                answer_start,
                answer_end,
                json_obj=json_obj,
                candidates=candidates,
            )

        logger.info(f"Total number of impossible answers: {impossible_answers}")

    def save(
        self,
        samples: Iterator[QASample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                json_object = sample.json_obj
                if use_predicted_annotation:
                    (
                        predicted_start_index,
                        predicted_end_index,
                    ) = sample.predicted_annotation
                    predicted_title = sample.context[
                        predicted_start_index:predicted_end_index
                    ]
                    json_object["output"] = [
                        {
                            "answer": predicted_title,
                            "provenance": [{"title": predicted_title}],
                        },
                        (predicted_start_index, predicted_end_index),
                    ]
                else:
                    json_object["output"].append((sample.char_start, sample.char_end))
                f.write(json.dumps(json_object) + "\n")


class BlinkDataDriver(QADataDriver):
    def __init__(self):
        self.pem_index = dict()
        self._load_pem_index()

    def _load_pem_index(self) -> None:
        with open(f"data/pem/pem.tsv") as f:
            for line in tqdm(f, desc="Loading pem index"):
                mention, *candidates = line.strip().split("\t")
                self.pem_index[mention] = candidates

    def read(self, lines: Iterator[str]) -> Generator[QASample, None, None]:

        impossible_answers = 0

        for line in lines:

            json_obj = json.loads(line.strip())
            meta_info = json_obj["meta"]

            # building question
            left_context = meta_info["left_context"]
            mention = meta_info["mention"]
            right_context = meta_info["right_context"]
            question = " ".join([left_context, "{", mention, "}", right_context])

            # building context
            candidates = self.pem_index.get(mention, [])

            if len(candidates) == 0:
                candidates = ["FAKE-CANDIDATE"]

            answer = json_obj["output"][0]["answer"]

            context, answer_start, answer_end = build_context(candidates, answer)

            if answer_start is None or answer_end is None:
                impossible_answers += 1

            yield QASample(
                context,
                question,
                answer_start,
                answer_end,
                json_obj=json_obj,
                candidates=candidates,
            )

        logger.info(f"Total number of impossible answers: {impossible_answers}")

    def save(
        self,
        samples: Iterator[QASample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        if use_predicted_annotation:
            raise ValueError(
                "Saving with predicted annotation on BlinkDataDriver is currently not supported"
            )
        with open(path, "w") as f:
            for sample in samples:
                sample.json_obj["output"].append((sample.char_start, sample.char_end))
                f.write(json.dumps(sample.json_obj))
                f.write("\n")


READERS_DICT[(QA, "aida")] = AidaDataDriver
READERS_DICT[(QA, "blink")] = BlinkDataDriver
READERS_DICT[(QA, "ed")] = AidaDataDriver
