import json
import os


class VirtualDB:
    def __init__(self, dbname: str = 'db'):
        self.fname = dbname + '.json'
        self.records = []
        if os.path.exists(self.fname):
            self.records = json.load(open(self.fname))

    def insert(self, record):
        if not record['id'] in [r['id'] for r in self.records]:
            self.records.append(record)
            json.dump(self.records, open(self.fname, 'w'))

    def select_all(self, column=None):
        if column:
            return [r[column] for r in self.records]
        return self.records
