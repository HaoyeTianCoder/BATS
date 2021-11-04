# Entrance
* **config.py** Set up your patch path(path_patch_root, path_generated_patch).  
* **main.py** Run it with parameters of RQ=['RQ1.1', 'RQ1.2', 'RQ2', 'RQ3.1', 'RQ3.2']

# Experiment


## RQ1.1

Customize: 
* **number:** premade the number of clusters. If more than 50, SSE Reduction Rate will be calculated as well.

OUTPUT: 
* **Qualified:** the ratio of clusters that have SC > 0 out of all clusters identified.
* **CSC:** the average value of similarity coefficient (SC) values for all clusters.
* **fig/RQ1/sc_clusters.png:** Similarity coefficient of test cases and patches at each cluster.

## RQ1.2

Customize: pass parameters option1 and option2.
* **option1:** Boolean. whether skip current project-id of search space(all projects or other projects).
* **option2:** Numerical. setup threshold for test cases similarity in scenario H.

OUTPUT: 
* **fig/RQ2/distribution_test_similarity.png:** Distribution on the similarities between each failing test case of each bug and its closest similar test case.
* **fig/RQ2/distribution_pairwise_patches.png:** Distributions on the similarities of pairwise patches.

## RQ2

Customize: Choose one of representation embeddings for patch under *config.py*.
* **cc2vec:** an attention-based model upon the AST representation of code method.
* **bert:** a transformer-based self-supervised model.
* **string:** raw strings(Levenshtein is used to calculate the distance of pairwise strings).

OUTPUT:
* **performance:** classification and ranking of the Baseline and BATS in terms of correct patches. 

## RQ3.1

Compare and enhance static approach

OUTPUT:
* **performance:** Comparison and enhancement of classification of the BATS and ML-based approach on patch correctness.

## RQ3.2

Compare and enhance dynamic approach

OUTPUT:
* **performance:** Comparison and enhancement of classification of the BATS and PatchSim on patch correctness.