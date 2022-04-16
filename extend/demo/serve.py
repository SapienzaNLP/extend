import argparse
import spacy
from pydantic import BaseModel, Field
from extend import spacy_component  # this is needed to register the spacy component

from typing import List

import uvicorn
from fastapi import FastAPI

from classy.utils.commons import get_local_ip_address
from classy.utils.log import get_project_logger

logger = get_project_logger(__name__)


def main(
    model_checkpoint_path: str,
    inventory_path: str,
    cuda_device: int,
    port: int,
):

    # load nlp
    nlp = spacy.load("en_core_web_sm")
    extend_config = dict(
        checkpoint_path=model_checkpoint_path,
        mentions_inventory_path=inventory_path,
        device=cuda_device,
        tokens_per_batch=10_000,
    )
    nlp.add_pipe("extend", after="ner", config=extend_config)

    # mock call to load resources
    nlp("Italy beat England and won Euro 2021.")

    # for better readability on the OpenAPI docs
    # why leak the inner confusing class names
    class Input(BaseModel):
        text: str = Field(None, description="Input text")

    class DisambiguatedEntity(BaseModel):
        char_start: int = Field(None, description="Start char index")
        char_end: int = Field(None, description="End char index")
        mention: str = Field(None, description="Mention")
        entity: str = Field(None, description="Disambiguated entity")

    class Output(Input):
        disambiguated_entities: List[DisambiguatedEntity] = Field(
            None, description="Disambiguated entities"
        )

    app = FastAPI(title="ExtEnD", version="1.0.0")

    @app.post("/", response_model=List[Output], description="Prediction endpoint")
    def predict(inputs: List[Input]) -> List[Output]:

        outputs = []

        for _input in inputs:
            _input = _input.text
            _doc = nlp(_input)
            _disambiguated_entities = []
            for _ent in _doc.ents:
                if _ent._.disambiguated_entity is not None:
                    _disambiguated_entities.append(
                        DisambiguatedEntity(
                            char_start=_ent.start_char,
                            char_end=_ent.end_char,
                            mention=_ent.text,
                            entity=_ent._.disambiguated_entity,
                        )
                    )
            outputs.append(
                Output(text=_input, disambiguated_entities=_disambiguated_entities)
            )

        return outputs

    @app.get("/healthz")
    def healthz():
        return "ok"

    local_ip_address = get_local_ip_address()
    print(f"Model exposed at http://{local_ip_address}:{port}")
    print(f"Remember you can checkout the API at http://{local_ip_address}:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("-p", type=int, help="Port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.model_path,
        "data/inventories/le-and-titov-2018-inventory.min-count-2.sqlite3",
        cuda_device=-1,
        port=args.p,
    )
