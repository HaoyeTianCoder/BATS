from code2vec.extractor import Extractor
import io
from os import path, mkdir
import tempfile

EXTRACTOR_JAR = "../representation/code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar"
MAX_CONTEXTS = 200
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2

class Code2vector:
    def __init__(self, model):
        self.model = model
        self.vector = None

    def convert(self, function):
        f = tempfile.NamedTemporaryFile(mode='w+', dir='/tmp', delete=True)
        f.write(function)
        file_path = f.name
        f.seek(0)
        extractor = Extractor(MAX_CONTEXTS, EXTRACTOR_JAR, MAX_PATH_LENGTH, MAX_PATH_WIDTH)
        paths, _ = extractor.extract_paths(file_path)
        f.close()
        result = self.model.predict(paths)

        if result:
            self.vector = result[0].code_vector
    
        return self.vector