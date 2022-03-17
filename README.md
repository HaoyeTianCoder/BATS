# Predicting Patch Correctness Based on the Similarity of Failing Test Cases

```bibtex
@article{tian2021checking,
  title={Checking Patch Behaviour against Test Specification},
  author={Tian, Haoye and Li, Yinghua and Pian, Weiguo and Kabor{\'e}, Abdoul Kader and Liu, Kui and Klein, Jacques and Bissyande, Tegawend{\'e} F},
  journal={arXiv preprint arXiv:2107.13296},
  year={2021}
}
```
Paper Link: https://arxiv.org/abs/2107.13296
# BATS
BATS, an unsupervised learning based system to predict patch correctness by checking patch **B**ehaviour **A**gainst failing **T**est **S**pecification.

### Preparation
* **data.**
  Unzip *PatchCollectingV1_sliced.zip* to obtain the generated patches by APR and Defects4j developer patches.
* **representation.**
  Follow *README.md* to download trained Code2Vec model.
* **other requirements.**
  Bert: cased_L-24_H-1024_A-16 
### Launch
* **experiment.** 
  Follow *experiment/README.md* to run RQ1, RQ2 and RQ3.
