# Representation
## Instructions for Code2Vec
1. Download the trained model and uncompress the file
```
wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz
tar -xvzf java14m_model.tar.gz
```
2. Update the variable `MODEL_MODEL_LOAD_PATH` in [./word2vector.py](https://github.com/HaoyeTianCoder/BATS/blob/main/representation/word2vector.py) according to destination folder of trained model 
