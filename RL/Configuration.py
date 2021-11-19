from baseline.utils import jsonParser


class Cfg:
    def __init__(self, path: str):
        parser = jsonParser(path)
        self.data = parser.loadParser()
        self.access_key: str = None
        self.secrete_key: str = None

        for key, value in self.data.items():
            setattr(self, key, value)