from typing import Optional, List, Iterator, Tuple, Dict, Union

import omegaconf
import torch
import torchmetrics
from classy.data.data_drivers import QASample

from classy.pl_modules.base import ClassyPLModule, ClassificationOutput

from classy.pl_modules.mixins.task import QATask
from transformers import AutoModelForQuestionAnswering, AutoConfig


class ESCModule(QATask, ClassyPLModule):
    def __init__(
        self,
        transformer_model: str,
        optim_conf: omegaconf.DictConfig,
        additional_special_tokens: Optional[List[str]] = None,
        attention_window: Optional[int] = None,
        modify_global_attention: Union[bool, int] = False,
    ):
        super().__init__(vocabulary=None, optim_conf=optim_conf)
        self.save_hyperparameters()

        if "longformer" in transformer_model:
            config = AutoConfig.from_pretrained(
                transformer_model, attention_window=attention_window
            )
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                transformer_model, config=config
            )
        else:
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                transformer_model
            )

        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            self.qa_model.resize_token_embeddings(
                self.qa_model.config.vocab_size + len(additional_special_tokens)
            )

        if type(modify_global_attention) == bool:
            modify_global_attention = 1 if modify_global_attention else 0
        self.modify_global_attention = modify_global_attention
        self.mode = "max-prod"
        self.evaluation = False

        # metrics
        self.start_accuracy_metric = torchmetrics.Accuracy()
        self.end_accuracy_metric = torchmetrics.Accuracy()
        self.accuracy_metric = torchmetrics.AverageMeter()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        start_position: Optional[torch.Tensor] = None,
        end_position: Optional[torch.Tensor] = None,
        candidates_offsets: Optional[List[Dict[str, Tuple[int, int]]]] = None,
        **kwargs,
    ) -> ClassificationOutput:

        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}

        if token_type_ids is not None:
            model_input["token_type_ids"] = token_type_ids

        if (
            "longformer" in self.hparams.transformer_model
            and self.modify_global_attention > 0
        ):
            if self.modify_global_attention == 1:
                global_attention = torch.zeros_like(attention_mask)
                global_attention[:, 0] = 1  # CLS global attention
                for i, co in enumerate(candidates_offsets):
                    for si, ei in co.values():
                        global_attention[i, si] = 1
                        global_attention[i, ei] = 1
            elif self.modify_global_attention == 2:
                global_attention = torch.zeros_like(attention_mask)
                first_candidate_starts = [
                    min([si for si, _ in cand_offs.values()])
                    for cand_offs in candidates_offsets
                ]
                for i, fcs in enumerate(first_candidate_starts):
                    global_attention[i, :fcs] = 1
            else:
                raise NotImplementedError
            model_input["global_attention_mask"] = global_attention

        if not self.evaluation:
            model_input["start_positions"] = start_position
            model_input["end_positions"] = end_position

        qa_output = self.qa_model(**model_input)

        packed_logits = torch.stack(
            [qa_output.start_logits, qa_output.end_logits], dim=0
        )
        packed_probabilities = torch.softmax(packed_logits, dim=-1)
        packed_predictions = torch.argmax(packed_logits, dim=-1)

        return ClassificationOutput(
            logits=packed_logits,
            probabilities=packed_probabilities,
            predictions=packed_predictions,
            loss=qa_output.loss,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        classification_output = self.forward(**batch)
        self.log("loss", classification_output.loss)
        return classification_output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        classification_output = self.forward(**batch)

        start_predictions = classification_output.predictions[0]
        end_predictions = classification_output.predictions[1]

        self.start_accuracy_metric(start_predictions, batch["start_position"])
        self.end_accuracy_metric(end_predictions, batch["end_position"])

        correct_full_predictions = torch.logical_and(
            torch.eq(start_predictions, batch["start_position"]),
            torch.eq(end_predictions, batch["end_position"]),
        )
        self.accuracy_metric(
            correct_full_predictions, torch.ones_like(correct_full_predictions)
        )

        self.log("val_loss", classification_output.loss)
        self.log("val_start_accuracy", self.start_accuracy_metric, prog_bar=True)
        self.log("val_end_accuracy", self.end_accuracy_metric, prog_bar=True)
        self.log("val_accuracy", self.accuracy_metric, prog_bar=True)

    def load_prediction_params(self, prediction_params: Dict):
        self.mode = prediction_params["mode"]
        self.evaluation = prediction_params["evaluation"]

    def select_indices(
        self,
        possible_indices: List[Tuple[int, int]],
        classification_probabilities: torch.FloatTensor,
    ) -> Tuple[int, int]:
        if self.mode == "max-prod":
            selector = (
                lambda x: classification_probabilities[0][x[0]]
                * classification_probabilities[1][x[1]]
            )
        elif self.mode == "max-end":
            selector = lambda x: classification_probabilities[1][x[1]]
        elif self.mode == "max-start":
            selector = lambda x: classification_probabilities[0][x[0]]
        elif self.mode == "max":
            selector = lambda x: max(
                classification_probabilities[0][x[0]],
                classification_probabilities[1][x[1]],
            )
        else:
            raise NotImplementedError
        return max(
            possible_indices,
            key=selector,
        )

    def batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token2chars: List[torch.Tensor],
        context_mask: torch.Tensor,
        samples: List[QASample],
        candidates_offsets: List[Dict[str, Tuple[int, int]]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Iterator[QASample]:

        classification_output = self.forward(
            input_ids,
            attention_mask,
            token_type_ids,
            candidates_offsets=candidates_offsets,
        )

        # search for best answer and yield
        for i in range(len(samples)):

            start_index, end_index = self.select_indices(
                possible_indices=list(candidates_offsets[i].values()),
                classification_probabilities=classification_output.probabilities[:, i],
            )

            samples[i].candidates_offsets = candidates_offsets[i]
            samples[i].token2chars = token2chars[i]

            try:
                start_index, end_index = (
                    token2chars[i][start_index][0].item(),
                    token2chars[i][end_index][1].item(),
                )
            except:
                start_index, end_index = -1, -1
            # yield

            samples[i].predicted_annotation = (start_index, end_index)

            yield samples[i]

    def read_input_from_bash(self) -> QASample:
        raise NotImplementedError(
            'You are likely doing "interactive predict", which is currently not supported in ExtEnD'
        )
