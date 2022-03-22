import argparse
import html
import time

from extend import spacy_component  # this is needed to register the spacy component

import spacy
import streamlit as st
from annotated_text import annotation
from classy.scripts.model.demo import tabbed_navigation
from classy.utils.streamlit import get_md_200_random_color_generator


def main(
    model_checkpoint_path: str,
    default_inventory_path: str,
    cuda_device: int,
):

    # setup examples
    examples = [
        "Italy beat England and won Euro 2021.",
        "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday.",
        "The project was coded in Java.",
    ]

    # css rules
    st.write(
        """
            <style type="text/css">
                a {
                    text-decoration: none !important;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

    # setup header
    st.markdown(
        "<h1 style='text-align: center;'>ExtEnD: Extractive Entity Disambiguation</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        """
            <div align="center">
                <a href="https://sunglasses-ai.github.io/classy/">
                    <img alt="Python" style="height: 3em; margin: 0 1em" src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gR2VuZXJhdG9yOiBBZG9iZSBJbGx1c3RyYXRvciAxNy4wLjAsIFNWRyBFeHBvcnQgUGx1Zy1JbiAuIFNWRyBWZXJzaW9uOiA2LjAwIEJ1aWxkIDApICAtLT4NCjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+DQo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxpdmVsbG9fMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeD0iMHB4IiB5PSIwcHgiDQoJIHdpZHRoPSI0MzcuNnB4IiBoZWlnaHQ9IjQxMy45NzhweCIgdmlld0JveD0iMCAwIDQzNy42IDQxMy45NzgiIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDQzNy42IDQxMy45NzgiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPHBhdGggZmlsbD0iI0ZGQ0MwMCIgZD0iTTM5NC42NjgsMTczLjAzOGMtMS4wMTgsMC0yLjAyLDAuMDY1LTMuMDE1LDAuMTUyQzM3NS42NjUsODIuODExLDI5Ni43OSwxNC4xNDYsMjAxLjgyMywxNC4xNDYNCgkJQzk1LjMxOSwxNC4xNDYsOC45OCwxMDAuNDg1LDguOTgsMjA2Ljk4OWMwLDEwNi41MDUsODYuMzM5LDE5Mi44NDMsMTkyLjg0MywxOTIuODQzYzk0Ljk2NywwLDE3My44NDItNjguNjY1LDE4OS44MjktMTU5LjA0NA0KCQljMC45OTUsMC4wODcsMS45OTcsMC4xNTIsMy4wMTUsMC4xNTJjMTguNzUxLDAsMzMuOTUyLTE1LjIsMzMuOTUyLTMzLjk1MkM0MjguNjIsMTg4LjIzOSw0MTMuNDE5LDE3My4wMzgsMzk0LjY2OCwxNzMuMDM4eg0KCQkgTTIwMS44MjMsMzQ2Ljg2OWMtNzcuMTMsMC0xMzkuODgtNjIuNzUtMTM5Ljg4LTEzOS44OGMwLTc3LjEyOSw2Mi43NS0xMzkuODc5LDEzOS44OC0xMzkuODc5czEzOS44OCw2Mi43NSwxMzkuODgsMTM5Ljg3OQ0KCQlDMzQxLjcwMywyODQuMTE5LDI3OC45NTMsMzQ2Ljg2OSwyMDEuODIzLDM0Ni44Njl6Ii8+DQoJPGc+DQoJCTxwYXRoIGZpbGw9IiNGRkNDMDAiIGQ9Ik0xMTQuOTA3LDIzMy40NzNjLTE0LjYyNiwwLTI2LjQ4My0xMS44NTYtMjYuNDgzLTI2LjQ4M2MwLTYyLjUyOCw1MC44NzEtMTEzLjQwMiwxMTMuMzk4LTExMy40MDINCgkJCWMxNC42MjYsMCwyNi40ODMsMTEuODU2LDI2LjQ4MywyNi40ODNzLTExLjg1NiwyNi40ODMtMjYuNDgzLDI2LjQ4M2MtMzMuMzI0LDAtNjAuNDMzLDI3LjExMi02MC40MzMsNjAuNDM2DQoJCQlDMTQxLjM5LDIyMS42MTcsMTI5LjUzNCwyMzMuNDczLDExNC45MDcsMjMzLjQ3M3oiLz4NCgk8L2c+DQo8L2c+DQo8L3N2Zz4NCg==">
                </a>
                <a href="https://spacy.io/" tyle="text-decoration: none">
                    <img alt="spaCy" style="height: 3em; margin: 0 1em;" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgOTAwIDUwMCAxNzUiIHdpZHRoPSIxNTAiIGhlaWdodD0iNTMiPjxwYXRoIGZpbGw9IiMwOUEzRDUiIGQ9Ik02NC44IDk3MC42Yy0xMS4zLTEuMy0xMi4yLTE2LjUtMjYuNy0xNS4yLTcgMC0xMy42IDIuOS0xMy42IDkuNCAwIDkuNyAxNSAxMC42IDI0LjEgMTMuMSAxNS40IDQuNyAzMC40IDcuOSAzMC40IDI0LjcgMCAyMS4zLTE2LjcgMjguNy0zOC43IDI4LjctMTguNCAwLTM3LjEtNi41LTM3LjEtMjMuNSAwLTQuNyA0LjUtOC40IDguOS04LjQgNS41IDAgNy41IDIuMyA5LjQgNi4yIDQuMyA3LjUgOS4xIDExLjYgMjEgMTEuNiA3LjUgMCAxNS4zLTIuOSAxNS4zLTkuNCAwLTkuMy05LjUtMTEuMy0xOS4zLTEzLjYtMTcuNC00LjktMzIuMy03LjQtMzQtMjYuNy0xLjgtMzIuOSA2Ni43LTM0LjEgNzAuNi01LjMtLjMgNS4yLTUuMiA4LjQtMTAuMyA4LjR6bTgxLjUtMjguOGMyNC4xIDAgMzcuNyAyMC4xIDM3LjcgNDQuOSAwIDI0LjktMTMuMiA0NC45LTM3LjcgNDQuOS0xMy42IDAtMjIuMS01LjgtMjguMi0xNC43djMyLjljMCA5LjktMy4yIDE0LjctMTAuNCAxNC43LTguOCAwLTEwLjQtNS42LTEwLjQtMTQuN3YtOTUuNmMwLTcuOCAzLjMtMTIuNiAxMC40LTEyLjYgNi43IDAgMTAuNCA1LjMgMTAuNCAxMi42djIuN2M2LjgtOC41IDE0LjYtMTUuMSAyOC4yLTE1LjF6bS01LjcgNzIuOGMxNC4xIDAgMjAuNC0xMyAyMC40LTI4LjIgMC0xNC44LTYuNC0yOC4yLTIwLjQtMjguMi0xNC43IDAtMjEuNSAxMi4xLTIxLjUgMjguMi4xIDE1LjcgNi45IDI4LjIgMjEuNSAyOC4yem01OS44LTQ5LjNjMC0xNy4zIDE5LjktMjMuNSAzOS4yLTIzLjUgMjcuMSAwIDM4LjIgNy45IDM4LjIgMzR2MjUuMmMwIDYgMy43IDE3LjkgMy43IDIxLjUgMCA1LjUtNSA4LjktMTAuNCA4LjktNiAwLTEwLjQtNy0xMy42LTEyLjEtOC44IDctMTguMSAxMi4xLTMyLjQgMTIuMS0xNS44IDAtMjguMi05LjMtMjguMi0yNC43IDAtMTMuNiA5LjctMjEuNCAyMS41LTI0LjEgMCAuMSAzNy43LTguOSAzNy43LTkgMC0xMS42LTQuMS0xNi43LTE2LjMtMTYuNy0xMC43IDAtMTYuMiAyLjktMjAuNCA5LjQtMy40IDQuOS0yLjkgNy44LTkuNCA3LjgtNS4xIDAtOS42LTMuNi05LjYtOC44em0zMi4yIDUxLjljMTYuNSAwIDIzLjUtOC43IDIzLjUtMjYuMXYtMy43Yy00LjQgMS41LTIyLjQgNi0yNy4zIDYuNy01LjIgMS0xMC40IDQuOS0xMC40IDExIC4yIDYuNyA3LjEgMTIuMSAxNC4yIDEyLjF6TTM1NCA5MDljMjMuMyAwIDQ4LjYgMTMuOSA0OC42IDM2LjEgMCA1LjctNC4zIDEwLjQtOS45IDEwLjQtNy42IDAtOC43LTQuMS0xMi4xLTkuOS01LjYtMTAuMy0xMi4yLTE3LjItMjYuNy0xNy4yLTIyLjMtLjItMzIuMyAxOS0zMi4zIDQyLjggMCAyNCA4LjMgNDEuMyAzMS40IDQxLjMgMTUuMyAwIDIzLjgtOC45IDI4LjItMjAuNCAxLjgtNS4zIDQuOS0xMC40IDExLjYtMTAuNCA1LjIgMCAxMC40IDUuMyAxMC40IDExIDAgMjMuNS0yNCAzOS43LTQ4LjYgMzkuNy0yNyAwLTQyLjMtMTEuNC01MC42LTMwLjQtNC4xLTkuMS02LjctMTguNC02LjctMzEuNC0uNC0zNi40IDIwLjgtNjEuNiA1Ni43LTYxLjZ6bTEzMy4zIDMyLjhjNiAwIDkuNCAzLjkgOS40IDkuOSAwIDIuNC0xLjkgNy4zLTIuNyA5LjlsLTI4LjcgNzUuNGMtNi40IDE2LjQtMTEuMiAyNy43LTMyLjkgMjcuNy0xMC4zIDAtMTkuMy0uOS0xOS4zLTkuOSAwLTUuMiAzLjktNy44IDkuNC03LjggMSAwIDIuNy41IDMuNy41IDEuNiAwIDIuNy41IDMuNy41IDEwLjkgMCAxMi40LTExLjIgMTYuMy0xOC45bC0yNy43LTY4LjVjLTEuNi0zLjctMi43LTYuMi0yLjctOC40IDAtNiA0LjctMTAuNCAxMS0xMC40IDcgMCA5LjggNS41IDExLjYgMTEuNmwxOC4zIDU0LjMgMTguMy01MC4yYzIuNy03LjggMy0xNS43IDEyLjMtMTUuN3oiIC8+IDwvc3ZnPg==">
                </a>
            </div> 
        """,
        unsafe_allow_html=True,
    )

    def model_demo():
        @st.cache(allow_output_mutation=True)
        def load_resources(inventory_path):

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
            nlp(examples[0])

            # return
            return nlp

        # read input
        placeholder = st.selectbox(
            "Examples",
            options=examples,
            index=0,
        )
        input_text = st.text_area("Input text to entity-disambiguate", placeholder)

        # custom inventory
        uploaded_inventory_path = st.file_uploader(
            "[Optional] Upload custom inventory (tsv file, mention \\t desc1 \\t desc2 \\t)",
            accept_multiple_files=False,
            type=["tsv"],
        )
        if uploaded_inventory_path is not None:
            inventory_path = f"data/inventories/{uploaded_inventory_path.name}"
            with open(inventory_path, "wb") as f:
                f.write(uploaded_inventory_path.getbuffer())
        else:
            inventory_path = default_inventory_path

        # load model and color generator
        nlp = load_resources(inventory_path)
        color_generator = get_md_200_random_color_generator()

        if st.button("Disambiguate", key="classify"):

            # tag sentence
            time_start = time.perf_counter()
            doc = nlp(input_text)
            time_end = time.perf_counter()

            # extract entities
            entities = {}
            for ent in doc.ents:
                if ent._.disambiguated_entity is not None:
                    entities[ent.start_char] = (
                        ent.start_char,
                        ent.end_char,
                        ent.text,
                        ent._.disambiguated_entity,
                    )

            # create annotated html components

            annotated_html_components = []

            assert all(any(t.idx == _s for t in doc) for _s in entities)
            it = iter(list(doc))
            while True:
                try:
                    t = next(it)
                except StopIteration:
                    break
                if t.idx in entities:
                    _start, _end, _text, _entity = entities[t.idx]
                    while t.idx + len(t) != _end:
                        t = next(it)
                    annotated_html_components.append(
                        str(annotation(*(_text, _entity, color_generator())))
                    )
                else:
                    annotated_html_components.append(str(html.escape(t.text)))

            st.markdown(
                "\n".join(
                    [
                        "<div>",
                        *annotated_html_components,
                        "<p></p>"
                        f'<div style="text-align: right"><p style="color: gray">Time: {(time_end - time_start):.2f}s</p></div>'
                        "</div>",
                    ]
                ),
                unsafe_allow_html=True,
            )

    def hiw():
        st.markdown("ExtEnD frames Entity Disambiguation as a text extraction problem:")
        st.image(
            "data/repo-assets/extend_formulation.png", caption="ExtEnD Formulation"
        )
        st.markdown(
            """            
            Given the sentence *After a long fight Superman saved Metropolis*, where *Superman* is the mention
            to disambiguate, ExtEnD first concatenates the descriptions of all the possible candidates of *Superman* in the
            inventory and then selects the span whose description best suits the mention in its context.
            
            To convert this task to end2end entity linking, as we do in *Model demo*, we leverage spaCy 
            (more specifically, its NER) and run ExtEnD on each named entity spaCy identifies 
            (if the corresponding mention is contained in the inventory).
        """
        )

    def abstract():
        st.write(
            """
            Local models for Entity Disambiguation (ED) have today become extremely powerful, in most part thanks to the advent of large pre-trained language models. However, despite their significant performance achievements, most of these approaches frame ED through classification formulations that have intrinsic limitations, both computationally and from a modeling perspective. In contrast with this trend, here we propose EXTEND, a novel local formulation for ED where we frame this task as a text extraction problem, and present two Transformer-based architectures that implement it. Based on experiments in and out of domain, and training over two different data regimes, we find our approach surpasses all its competitors in terms of both data efficiency and raw performance. EXTEND outperforms its alternatives by as few as 6 F 1 points on the more constrained of the two data regimes and, when moving to the other higher-resourced regime, sets a new state of the art on 4 out of 6 benchmarks under consideration, with average improvements of 0.7 F 1 points overall and 1.1 F 1 points out of domain. In addition, to gain better insights from our results, we also perform a fine-grained evaluation of our performances on different classes of label frequency, along with an ablation study of our architectural choices and an error analysis. We release our code and models for research purposes at https:// github.com/SapienzaNLP/extend.
            
            Link to full paper: https://www.researchgate.net/publication/359392427_ExtEnD_Extractive_Entity_Disambiguation 
        """
        )

    tabs = dict(
        model=("Model demo", model_demo),
        hiw=("How it works", hiw),
        abstract=("Abstract", abstract),
    )

    tabbed_navigation(tabs, "model")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model_path, "data/inventories/aida.tsv", cuda_device=-1)
