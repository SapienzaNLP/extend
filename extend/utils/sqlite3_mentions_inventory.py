import sqlite3
from typing import Dict


class Sqlite3BackedMentionsInventory:
    @staticmethod
    def create(mentions_inventory: Dict, output_path: str):
        connection = sqlite3.connect(output_path)
        cursor = connection.cursor()
        # create schema
        cursor.execute(
            """
                CREATE TABLE candidates (mention text PRIMARY KEY, entities text)
            """
        )
        # insert
        inserts = []
        for k, v in mentions_inventory.items():
            inserts.append((k, "\t".join(v)))
        cursor.executemany("INSERT INTO candidates VALUES (?, ?)", inserts)
        # commit and close
        connection.commit()
        connection.close()

    @classmethod
    def from_path(cls, path: str):
        return cls(sqlite3.connect(path, check_same_thread=False))

    def __init__(self, connection):
        self.connection = connection
        self.cursor = self.connection.cursor()

    def __getitem__(self, key: str):
        self.cursor.execute("SELECT * FROM candidates where mention=(?)", (key,))
        result = self.cursor.fetchone()
        if result is None:
            raise KeyError(key)
        candidates = result[1]
        return candidates.split("\t")

    def __contains__(self, key: str):
        self.cursor.execute("SELECT * FROM candidates where mention=(?)", (key,))
        result = self.cursor.fetchone()
        return result is not None

    def __len__(self):
        self.cursor.execute("SELECT * FROM candidates")
        return len(self.cursor.fetchall())


if __name__ == "__main__":
    print(
        len(Sqlite3BackedMentionsInventory.from_path("data/inventories/aida.sqlite3"))
    )
    print(
        len(
            Sqlite3BackedMentionsInventory.from_path(
                "data/inventories/le-and-titov-2018-inventory.min-count-2.sqlite3"
            )
        )
    )
    print(
        len(
            Sqlite3BackedMentionsInventory.from_path(
                "data/inventories/le-and-titov-2018-inventory.sqlite3"
            )
        )
    )
