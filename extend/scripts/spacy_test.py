def main():

    import spacy
    from extend import spacy_component  # this is needed to register the spacy component

    nlp = spacy.load("en_core_web_sm")
    extend_config = dict(
        checkpoint_path="experiments/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt",
        mentions_inventory_path="data/inventories/le-and-titov-2018-inventory.min-count-2.sqlite3",
        device=0,
        tokens_per_batch=10_000,
    )
    nlp.add_pipe("extend", after="ner", config=extend_config)

    input_sentence = "Japan began the defence of their title with a lucky 2-1 win against Syria in a championship match on Friday."

    doc = nlp(input_sentence)
    for ent in doc.ents:
        if ent._.disambiguated_entity is not None:
            print(
                f"Mention: {ent.text} | Entity type: {ent.label_} | Wikipedia Page Title: {ent._.disambiguated_entity}"
            )


if __name__ == "__main__":
    main()
