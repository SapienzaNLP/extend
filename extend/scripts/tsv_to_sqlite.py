import argparse
import sqlite3
from pathlib import Path

from tqdm import tqdm

from extend.spacy_component import load_mentions_inventory
from extend.utils.sqlite3_mentions_inventory import Sqlite3BackedMentionsInventory


def main():
    args = parse_args()
    input_path, output_path = args.input_path, args.output_path
    assert input_path.endswith(".tsv") and output_path.endswith(".sqlite3")
    assert not Path(output_path).exists()

    # read input inventory
    inventory_stores = load_mentions_inventory(input_path)

    # convert it to sql
    Sqlite3BackedMentionsInventory.create(inventory_stores, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    return parser.parse_args()


if __name__ == "__main__":
    main()
