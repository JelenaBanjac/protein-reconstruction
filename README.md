# Learning to recover orientations from projections in single-particle cryo-EM

[Jelena Banjac](https://jelenabanjac.com), Data Science master student,
[Laurène Donati](https://people.epfl.ch/laurene.donati), BIG,
[Michaël Defferrard](https://deff.ch/), LTS2, EPFL.

* Paper: [`arXiv:2104.06237`](https://arxiv.org/abs/2104.06237)
* Website with interactive visualizations: <https://jelenabanjac.com/protein-reconstruction/home.html>

> A major challenge in single-particle cryo-electron microscopy (cryo-EM) is that the orientations adopted by the 3D particles prior to imaging are unknown; yet, this knowledge is essential for high-resolution reconstruction.
> We present a method to recover these orientations directly from the acquired set of 2D projections.
> Our approach consists of two steps: (i) the estimation of distances between pairs of projections, and (ii) the recovery of the orientation of each projection from these distances.
> In step (i), pairwise distances are estimated by a Siamese neural network trained on synthetic cryo-EM projections from resolved bio-structures.
> In step (ii), orientations are recovered by minimizing the difference between the distances estimated from the projections and the distances induced by the recovered orientations.
> We evaluated the method on synthetic cryo-EM datasets.
> Current results demonstrate that orientations can be accurately recovered from projections that are shifted and corrupted with a high level of noise.
> The accuracy of the recovery depends on the accuracy of the distance estimator.
> While not yet deployed in a real experimental setup, the proposed method offers a novel learning-based take on orientation recovery in SPA.

![two-step method](images/schematic_method_overview-1.jpg)

## Repository content

[Notebooks](./notebooks), used to reproduce our findings, are divided in the following phases:

0. [Data preparation](https://jelenabanjac.com/protein-reconstruction/phase0_intro.html): generate 2D projections from a protein
1. [Distance estimation](https://jelenabanjac.com/protein-reconstruction/phase1_intro.html): learn a function to estimate the distance between two projections
2. [Orientation recovery](https://jelenabanjac.com/protein-reconstruction/phase2_intro.html): recover the projections' orientations from estimated distances
3. [Protein reconstruction](https://jelenabanjac.com/protein-reconstruction/phase3_intro.html): reconstruct the protein from its projections and their recovered orientations

The notebooks in each folder represent different experimental conditions or modeling approach.

Additionally, the [`cryoem`](./cryoem) python package contains scripts to generate a huge amount of 2D projections with corresponding orientation.

## Installation

First, download and install Anaconda or Miniconda on your machine, link [here](https://www.anaconda.com/products/individual). Note: the project was developed with Python 3.6+.

Then open the terminal and type following:
```bash
# clone the repo
git clone https://github.com/JelenaBanjac/protein-reconstruction.git

# position yourself inside the project
cd protein-reconstruction

# create environment
conda env create -f environment.yml

# activate environment
conda activate protein_reconstruction
```
Now you are able to use the code and run the notebooks you wish!

[Optional] Test if some dependencies are installed:
```bash
# tensorflow check
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# o/w install with: 
pip3 install tensorflow-gpu
pip3 install tensorflow-graphics

# astra toolbox check
python3 -c "import astra;astra.test_CUDA()"
```

To run the jupyter notebooks (`$1` is GPU id, `$2` is port for jupyter notebook if ran externally):

```bash
cd $HOME/protein-reconstruction/notebooks
source activate protein_reconstruction
export CUDA_VISIBLE_DEVICES=$1
nohup $HOME/miniconda/envs/protein_reconstruction/bin/jupyter notebook --ip=0.0.0.0 --port=$2 &
```

For more information how to do use the pachage methods, checkout the [website](https://jelenabanjac.com/protein-reconstruction/home.html) with the example.

## Package versions
The following versions of the packages are used in the project.
```
python-3.6.8
cudnn-7.1.3
cudatoolkit-8.0
```
```
cuda 10
cuDNN 7
nvidia driver 415
```

```
$ nvcc --version
$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

## Licence & citation

The code in this repository is released under the terms of the [MIT license](LICENSE).
Please cite our paper if you use it.

```
@inproceedings{cryoem_orientation_recovery,
  title = {Learning to recover orientations from projections in single-particle cryo-EM},
  author = {Banjac, Jelena, Donati, Laur\`ene, and Defferrard, Micha\"el},
  year = {2021},
  archivePrefix={arXiv},
  eprint={2104.06237},
  url = {https://arxiv.org/abs/2104.06237},
}
