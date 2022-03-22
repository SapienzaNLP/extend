from typing import List, Tuple, Dict

from classy.data.data_drivers import QASample
from classy.evaluation.base import Evaluation


class InKBF1(Evaluation):
    def __call__(self, path: str, predicted_samples: List[QASample]) -> Dict:

        gold_indices = [
            qa_sample.reference_annotation for qa_sample in predicted_samples
        ]
        predicted_indices = [
            qa_sample.predicted_annotation
            if qa_sample.candidates[0] != "FAKE-CANDIDATE"
            else "NIL"
            for qa_sample in predicted_samples
        ]

        tp = 0
        inkb_predictions = 0
        total_predictions = 0

        for pi, gi in zip(predicted_indices, gold_indices):
            total_predictions += 1
            if pi != "NIL":
                inkb_predictions += 1
                if pi == gi:
                    tp += 1

        precision = tp / inkb_predictions
        recall = tp / total_predictions
        f1 = 2 * precision * recall / (precision + recall)

        return dict(precision=precision, recall=recall, f1=f1)
