# Predicting Patch Correctness Based on the Similarity of Failing Test Cases

```bibtex
@article{tian2022predicting,
  title={Predicting Patch Correctness Based on the Similarity of Failing Test Cases},
  author={Tian, Haoye and Li, Yinghua and Pian, Weiguo and Kabore, Abdoul Kader and Liu, Kui and Habib, Andrew and Klein, Jacques and Bissyand{\'e}, Tegawend{\'e} F},
  journal={ACM Transactions on Software Engineering and Methodology},
  year={2022},
  publisher={ACM New York, NY},
  url = {https://doi-org.proxy.bnl.lu/10.1145/3511096},
  doi = {10.1145/3511096}
}
```

# BATS
BATS, an unsupervised learning based system to predict patch correctness by checking patch **B**ehaviour **A**gainst failing **T**est **S**pecification.

## Ⅰ) Requirements
* **Data.**
Download the BATS_Dataset from [data in Zenodo](https://zenodo.org/record/7020346#.YwaKCGQzZb8). 
Set up `self.path_generated_patch` in **experiment/config.py** with the path of the downloaded *PatchCollectingV1_sliced*.

* **Code2Vec representation model.**
  1. Download the trained model and uncompress the file.
  `wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz tar -xvzf java14m_model.tar.gz`
  2. Update the variable `MODEL_MODEL_LOAD_PATH` in [./word2vector.py](https://github.com/HaoyeTianCoder/BATS/blob/main/representation/word2vector.py) according to destination folder of trained model 

* **BERT model.**
    * BERT model client&server: 24-layer, 1024-hidden, 16-heads, 340M parameters. download it [here](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip).
    * Environment for BERT server (different from reproduction)
      * python 3.7 
      * pip install tensorflow==1.14
      * pip install bert-serving-client==1.10.0
      * pip install bert-serving-server==1.10.0
      * pip install protobuf==3.20.1
      * Launch BERT server via `bert-serving-start -model_dir "Path2BertModel"/wwm_cased_L-24_H-1024_A-16 -num_worker=2 -max_seq_len=360`

## Ⅱ) Reproduction
  Follow the [experiment/README.md](https://github.com/HaoyeTianCoder/BATS/blob/main/experiment/README.md) to obtain the experimental results in the paper.

## Ⅲ) Custom Prediction 
To predict the correctness of your custom patches, you are welcome to use the prediction interface.
```
python main.py predict $cut-off $bug_id $path2patch
```

For instance: 
```
python main.py predict 0.8 Chart_26 "path2dataset"/BATS_DataSet/PatchCollectingV1_sliced/PraPR/Correct/Chart/26/patch1
```
