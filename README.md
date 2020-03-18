<p align="center">
	<img src="https://github.com/daiquocnguyen/Caps2NE/blob/master/caps2ne_logo.png">
</p>

## A Capsule Network-based Model for Learning Node Embeddings<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FCaps2NE%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/Caps2NE"><a href="https://github.com/daiquocnguyen/Caps2NE/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/Caps2NE"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/Caps2NE">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/Caps2NE">
<a href="https://github.com/daiquocnguyen/Caps2NE/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/Caps2NE"></a>
<a href="https://github.com/daiquocnguyen/Caps2NE/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/Caps2NE"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/Caps2NE">

This program provides the implementation of our unsupervised node embedding model Caps2NE as described in [the paper](https://arxiv.org/pdf/1911.04822.pdf):

        @InProceedings{Nguyen2019Caps2NE,
          author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dat Quoc Nguyen and Dinh Phung},
          title={{A Capsule Network-based Model for Learning Node Embeddings}},
          booktitle={arXiv:1911.04822v1},
          year={2019}
          }
  
<p align="center">
	<img src="https://github.com/daiquocnguyen/Caps2NE/blob/master/Caps2NE.png">
</p>

## Usage

### Requirements
- Python 3.x
- Tensorflow 1.x
- scikit-learn

### Training
To run the program in the transductive setting:

	$ python train_Caps2NE.py --embedding_dim <int> --name <dataset_name> --run_folder <name_of_running_folder> --num_epochs <int> --batch_size <int> --num_sampled <int> --learning_rate <float> --iter_routing <int> --model_name <name_of_saved_model>

To run the program in the inductive setting for Cora/Citeseer/Pubmed:

	$ python train_Caps2NE_ind.py --embedding_dim <int> --nameTrans <data_name_trans> --nameInd <dat_name_ind> --idx_time <int> --run_folder <name_of_running_folder> --num_epochs <int> --batch_size <int> --iter_routing <int> --num_sampled <int> --learning_rate <float> --model_name <name_of_saved_model>

	
**Parameters:** 

`--embedding_dim`: The embedding size (and used as the dimension size of feature vectors for a random initialization in case of POS/PPI/BlogCatalog).

`--learning_rate`: The initial learning rate for the Adam optimizer.

`--model_name`: Name of saved model.

`--run_folder`: Specify directory path to save trained models.

`--batch_size`: The batch size.

`--num_sampled`: The number of samples for the sampled softmax loss function.

`--iter_routing`: The number of dynamic routing iterations. 

`--num_epochs`: The number of training epochs.

`--idx_time`: The index time of sampling training/validation/test sets (in {1, 2, ..., 10}).

**Command examples:**
		
	$ python train_Caps2NE.py --embedding_dim 128 --name cora.128.10.trans.pickle --batch_size 64 --num_sampled 256 --iter_routing 1 --learning_rate 0.00005 --model_name cora_trans_iter1_3

	$ python train_Caps2NE_ind.py --embedding_dim 128 --nameTrans citeseer.128.10.trans.pickle --nameInd citeseer.128.10.ind1.pickle --idx_time 1 --batch_size 64 --iter_routing 1 --num_sampled 256 --learning_rate 0.00005 --model_name citeseer_ind1_3

	
In case the OS system installed `sbatch`, you can run a command: `sbatch file_name.script`, see script examples in the folder `scripts`. 

### Evaluation

You can modify `scoring.py` at https://github.com/phanein/deepwalk/tree/master/example_graphs to evaluate the learned node embeddings for the node classification task on POS/PPI/BlogCatalog, with using a 10-folds-cross-validation on the training set for each fraction value to find optimal hyper-parameters for each setup.

**Command examples:**

	$ python scoring_transductive.py --input cora --output cora --tmpString cora

	$ python scoring_inductive.py --input _ind1_ --output cora --idx_time 1 --tmpString cora

### Notes

It is important to note on Cora/Citeseer/Pubmed that you should report the mean and standard deviation of accuracies over 10 different times of sampling training/validation/test sets when comparing models.

File `utils.py` has a function `sampleUniformRand` to randomly sample 10 different data splits of training/validation/test sets. I also include my 10 different data splits in `dataset_name.10sampledtimes`.

In some preliminary experiments, you can use the process of sampling 32 random walks (instead of 128) for each node which gets similar performances, for a faster training on Cora/Citeseer/Pubmed.

File `sampleRWdatasets.py` is used to generate random walks. See command examples in `commands.txt`:
		
	$ python sampleRWdatasets.py --input graph/cora.Full.edgelist --output graph/cora.128.10.trans.pickle
		
	$ python sampleRWdatasets.py --input data/pubmed.ind.edgelist1 --output graph/pubmed.128.10.ind1.pickle

Unzip the zipped files in the folder `graph`.

## License

Please cite the paper whenever Caps2NE is used to produce published results or incorporated into other software. As a free open-source implementation, Caps2NE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

Caps2NE is  is licensed under the Apache License 2.0.

