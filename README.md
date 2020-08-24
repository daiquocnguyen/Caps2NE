<p align="center">
	<img src="https://github.com/daiquocnguyen/Caps2NE/blob/master/caps2ne_logo.png">
</p>

# A Capsule Network-based Model for Learning Node Embeddings<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FCaps2NE%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/Caps2NE"><a href="https://github.com/daiquocnguyen/Caps2NE/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/Caps2NE"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/Caps2NE">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/Caps2NE">
<a href="https://github.com/daiquocnguyen/Caps2NE/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/Caps2NE"></a>
<a href="https://github.com/daiquocnguyen/Caps2NE/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/Caps2NE"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/Caps2NE">

This program provides the implementation of our unsupervised node embedding model Caps2NE as described in [our paper](https://arxiv.org/pdf/1911.04822.pdf) where we use a capsule network to unsupervisedly learn embeddings of nodes in generated random walks.
  
<p align="center">
	<img src="https://github.com/daiquocnguyen/Caps2NE/blob/master/Caps2NE.png" width="550">
</p>

## Usage

### Requirements
- Python 3.x
- Tensorflow 1.x
- scikit-learn

### Training
Regarding the transductive setting:

	$ python train_Caps2NE.py --embedding_dim 128 --name cora.128.10.trans.pickle --batch_size 64 --num_sampled 256 --iter_routing 1 --learning_rate 0.00005 --model_name cora_trans_iter1_3

Regarding the inductive setting:

	$ python train_Caps2NE_ind.py --embedding_dim 128 --nameTrans citeseer.128.10.trans.pickle --nameInd citeseer.128.10.ind1.pickle --idx_time 1 --batch_size 64 --iter_routing 1 --num_sampled 256 --learning_rate 0.00005 --model_name citeseer_ind1_3

### Notes

File `utils.py` has a function `sampleUniformRand` to randomly sample 10 different data splits of training/validation/test sets. I also include my 10 different data splits in `dataset_name.10sampledtimes`. You can see command examples in `commands.txt`.

## Cite

Please cite the paper whenever Caps2NE is used to produce published results or incorporated into other software:
	
	@InProceedings{Nguyen2020Caps2NE,
          author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dat Quoc Nguyen and Dinh Phung},
          title={{A Capsule Network-based Model for Learning Node Embeddings}},
          booktitle={The 29th ACM International Conference on Information and Knowledge Management (CIKM2020)},
          year={2020}
          }

## License

As a free open-source implementation, Caps2NE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

Caps2NE is  is licensed under the Apache License 2.0.

