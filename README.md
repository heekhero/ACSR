# Worst Case Matters for Few-Shot Recognition

This repository is the official implementation of [Worst Case Matters for Few-Shot Recognition](https://arxiv.org/abs/2203.06574), which is accepted by ECCV2022. 

This paper boost the worst-case performance in few-shot learning by reducing the standard deviation and increasing the average accuracy simultaneously from the perspective of bias-variance trade-off. To achieving so, a simple yet effective stability regularization (SR) loss together with model ensemble to reduce variance during fine-tuning, and an adaptability calibration (AC) mechanism to reduce the bias are proposed.

### Environment
 - Python 3.8

 - [Pytorch](http://pytorch.org/) 1.10.0

 - Pandas
   


### Running the code


***Datasets***

Download datasets from [this link](https://drive.google.com/drive/folders/1ey17LD4bbGJUhWtj2Jb7DsePyArn6gnF?usp=sharing), and organize these datasets as follows

```
YOUR_DATA_PATH
	miniImagenet/
		base/
		val/
		novel/
	CUB/
		base/
		val/
		novel/
	CIFAR-FS/
		base/
		val/
		novel/
```

After that, modify the parameter 'DATA_PATH' in **config.py** with the path of YOUR_DATA_PATH.



***Checkpoints***

Download file **checkpoint.tar.gz** ([link](https://drive.google.com/drive/folders/1ey17LD4bbGJUhWtj2Jb7DsePyArn6gnF?usp=sharing)), unpack it into a directory named **checkpoint** and place it in the root path of the project. The checkpoint contains everything else you need to run the code, including pre-trained models, pre-sampled tasks/episodes and features of base set extracted by pre-trained models.



 We highly encourage researchers to conduct experiments on our pre-sampled episodes to further explore the way to boost few-shot learning, especially for worst-case performance.



**Train**

For AC+SR

```
python meta_test_ACSR.py --dataset [DATASET] --k [SHOT] 
```

For AC+EnSR, thanks to the parallel computing capabilities of pytorch, please run the code as

```
CUDA_VISIBLE_DEVICES=[CUDA_INDICES, e.g. 0,1,2,3] python -m torch.distributed.launch --nproc_per_node [NUM_OF_NODES, e.g. 4] --master_port [PORT]   meta_test_EnACSR.py --dataset [DATASET] --k [SHOT] 
```

which naturally allows the ensemble methods to run in parallel.



**Important Arguments**
Some important arguments for our code.

- `dataset`: choices=['miniImagenet', 'CUB', 'CIFAR-FS']
- `arch`: choices=['wrn_28_10']
- `lr`: learning rate of fine-tuning on each task.
- `tune_mode` : adaptation calibration (AC) strategy used to the model. By default, the last convolutional block and final classification layer is set to be learnable.
- `beta`: trade-off scalar for classification loss and stability regularization (SR) loss.
- `aux_bs`: batch size to compute SR loss in one step.
- `epochs_finetune`: epochs for fine-tuning each few-shot episode.
- `n` : how many ways per few-shot episode.
- `k` : how many shots per few-shot episode.
- `q` : how many queries per few-shot episode.
- `tasks` : the number of episodes for evaluation..



### Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@article{fu2022ACSR,
  title={Worst Case Matters for Few-Shot Recognition},
  author={Fu, Minghao and Cao, Yun-Hao and Wu, Jianxin},
  journal={The European Conference on Computer Vision},
  year={2022}
}
```