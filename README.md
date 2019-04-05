## **gvnr**: python package for the paper "Global Vectors for Node Representation" (WWW19)
https://arxiv.org/pdf/1902.11004.pdf

### To run the experiments presented in the paper:

You need python 3.6 installed on your computer with pip. Optionally create a new environment (with conda):
    
    conda create --name gvnr python=3.6 pip
    
Then run:

    git clone https://github.com/brochier/gvnr
    cd gvnr
    pip install -r requirements.txt 
    python scripts/eval.py -h
    
The file `scripts/eval.py` accepts the arguments *--save* (to export the embeddings) *--method* (to select a particular algorithm) and *--dataset* (to select a particular dataset).


### To install gvnr as a package:

pip install the repository:

    pip install git+git://github.com/brochier/gvnr.git
    
The file `scripts/example.py` provides an example how to use the package:

```python
"""

An example script to use gvnr as a package. Its shows how to load and process a
dataset, setup the GVNR-t algorithm, train it and evaluate it.
"""
import os
import sys
import logging
import gvnr.data.datasets
import gvnr.evaluation
import gvnr.preprocessing.random_walker
import gvnr.preprocessing.window_slider
import gvnr.models.gvnrt
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse


# We will use the logging package with DEBUG log level
logging.basicConfig(level=logging.DEBUG,
                    format="[%(levelname)s] [%(asctime)s] %(message)s",
                    datefmt="%y-%m-%d %H:%M:%S")
logger = logging.getLogger()


logger.info("1) LOADING A DATASET")
"""
We load the dataset "cora" with following numpy.ndarray-s:
binary_vectors:     binary bags of words of the documents of the network (only for "cora" and "citeseer")
tfidf_vectors:      tf-idf weighted bags of words of the documents of the network (only for "cora" and "citeseer")      
svd_vectors:        svd vectors obtained from previous tf-idf vectors
adjacency_matrix:   adjacency matrix of the network
labels:             groundtruth labels for classification
gt_mask:            mask to select nodes that are linked with a label (only useful with aminer)
"""
binary_vectors, tfidf_vectors, svd_vectors, adjacency_matrix, labels, gt_mask = gvnr.data.datasets.get_dataset("cora")


logger.info("2) PREPROCESSING THE DATA")
"""
We perform random walks on the network. N being the number of nodes, we get N*10 sequences of nodes of lengths 40.
These are stored in a numpy.ndarray  
"""
random_walker = gvnr.preprocessing.random_walker.RandomWalker(
                scipy.sparse.csr_matrix(adjacency_matrix),
                walks_length=40,
                walks_number=10
            )
random_walks = random_walker.build_random_walks()
"""
From these sequences of nodes, we slide a window to increment a matrix of counts of co-occurring nodes. We look 5
context nodes on the left and on the right of a target node. The window_factor is chosen such that co-occurrence counts
are decreasingly incremented given the distance to the target node e.g:
[1/5, 1/4, 1/3, 1/2, 1, target_node, 1, 1/2, 1/3, 1/4, 1/5] 
"""
slider = gvnr.preprocessing.window_slider.WindowSlider(
    random_walks,
    adjacency_matrix.shape[0],
    window_size=5,
    window_factor="decreasing"
)
X = slider.build_cooccurrence_matrix()


logger.info("3) TRAINING THE MODEL")
"""
We load the model GVNR-T, implemented with tensorflow. We provide several parameters:
param1:              co-occurrence counts of the nodes.
param2:              textual features of the nodes, as sparse weighted bag of words.
learn_rate:          learning rate for the ADAM optimizer.
embedding_size:      embedding dimension.
batch_size:          number of pairs of nodes given at each iteration before updating the parameters.
n_epochs:            number of time we go through the entire X matrix.
k_neg:               number of negative samples to draw for each co-occurrence.
x_min:               minimum co-occurrence count value to consider (filtering X).                 
"""
model = gvnr.models.gvnrt.gvnrt()
model.fit(X,
          scipy.sparse.csr_matrix(tfidf_vectors),
          learn_rate=0.003,
          embedding_size=128,
          batch_size=128,
          n_epochs=2,  # too low
          k_neg=1,
          x_min=10  # too high
          )
# We get the trained embeddings and normalize them before evaluation.
I = normalize(model.get_embeddings(embedding='I'), norm='l2')
J = normalize(model.get_embeddings(embedding='J'), norm='l2')
vectors = normalize(np.hstack([I, J]), axis=0, norm='l2')


logger.info("4) EVALUATION")
"""
To evaluate the vectors, we perform a classification task. With cora, there is only 1 label per node. We train a linear
classifier with 20% and 80% of the data for training with 5 trials where training data are randomly sampled.  
"""
training_proportions = [0.2, 0.8]
scores = gvnr.evaluation.get_score(
    vectors[gt_mask], # gt_mask only needed for aminer
    labels,
    multilabels=False,
    proportions=training_proportions,
    n_trials = 5
)
logger.info("DATASET: {0}, BASELINE: {1} => f1_micro {2} f1_macro {2}".format("Cora", "GVNR-t", training_proportions))
logger.info("   ".join(["&{0:.1f}".format(s*100) for s in list(scores["f1_micro"]) + list(scores["f1_macro"]) ] ))
```



### Citation

If you use this code, please consider citing the paper:

    @inproceedings{brochier2019global,
      title={Global Vectors for Node Representations},
      author={Brochier, Robin and Guille, Adrien and Velcin, Julien},
      booktitle={Proceedings of the 2019 World Wide Web Conference on World Wide Web},
      year={2019},
      organization={International World Wide Web Conferences Steering Committee},
    }
    
