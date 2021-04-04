# Protein 3D Poses Recovery

**3D Poses Recovery in Single-Particle Cryo-EM from Learned Pairwise Projection Distances**

The topic of this project is to learn pairwise projection distances in order to recover the angles at which we imaged these 2D projections from a given 3D protein.

## Summary
Single-particle cryo-electron microscopy (cryo-EM) is a technology that allows the observation and the high-resolution 3D structure determination of biomolecules. In this project, the goal is to estimate the angles at which we imaged the 2D projections from a given 3D protein (cf illustration bellow). We developed deep learning models to estimate the angles from learned pairwise projection distances. We designed a two-step method: 1) **distance estimation** using a Siamese neural network to learn the distance between pairs of projections, and 2) **angle recovery** that includes a minimization scheme in order to estimate the angles at which each projection was taken. The current results obtained are discussed depending on different combination of approaches used andexperimental conditions.
![images/spcryoem.png](images/spcryoem.png)

## General Flow
General flow of the project can be seen in the illustration bellow:
![images/protein_flow.png](images/protein_flow.png)

## Report
More details on the implementation can be found in the [report](reports/Report_BIGSemesterProject_JelenaBanjac.pdf).  
The presentation slideshow can be found on this [link](https://docs.google.com/presentation/d/e/2PACX-1vSeN_Zd4mL9ScdvlEAIib4QFq3kkUxojnj-YBEAGuxKxPDQ48PCL2Y_JBT4cn_UBcIFhPp_YnNZZF1c/pub?start=true&loop=false&delayms=3000) and the presentation material on this [link](reports/Presentation_BIGSemesterProject_JelenaBanjac.pdf).

## Repository
This repository contains scripts to generate a huge amount of 2D projections with corresponding angles of 3D volumes. 
Also, it contains the notebooks with different combinations of project approaches.

## Installation
First, download and install Anaconda on your machine, link [here](https://www.anaconda.com/products/individual). Note: the project was developed with Python 3.6+.

Then open the terminal and type following:
```
# clone the repo
$ git clone https://github.com/JelenaBanjac/protein-reconstruction.git

# position yourself inside the project
$ cd protein-reconstruction

# create environment
$ conda env create -f environment.yml

# activate environment
$ conda activate protein_reconstruction
```
Now you are able to use the code and run the notebooks you wish!

[Optional] Test if some dependencies are installed:
```
# tensorflow check
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# o/w install with: 
$ pip3 install tensorflow-gpu
$ pip3 install tensorflow-graphics

# astra toolbox check
$ python3 -c "import astra;astra.test_CUDA()"
```

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

## Logbook
Notes taken during the project development:
- [Experiments notes](https://app.box.com/s/8heyh18d473xetiqzu1eorkzk29ax40e)
- [Meeting notes](https://app.box.com/s/0x42ke3j5e6yyoomlhukcayf4qz3ezgw)
- [Summary notes of work left to be done after first 4 months](https://app.box.com/s/ndgnxrgompchlhr7o2hoqaalacjp98hd)

## Team
**Student:**  
[Jelena Banjac](https://jelenabanjac.com), jelena.banjac@epfl.ch, Data Science Master Student

**Supervisors:**  
[Laurène Donati](https://people.epfl.ch/laurene.donati?lang=en), laurene.donati@epfl.ch, BIG, EPFL  
[Michaël Defferrard](https://deff.ch/), michael.defferrard@epfl.ch, LTS2, EPFL

**Professor:**  
[Michaël Unser](http://bigwww.epfl.ch/unser/), michael.unser@epfl.ch, BIG, EPFL

