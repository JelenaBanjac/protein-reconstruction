# 3D Poses Recovery in Single-Particle Cryo-EM from Learned Pairwise Projection Distances

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

## Run

```
usage: generator.py [-h] --config-file CONFIG_FILE [--input-file INPUT_FILE]
                    [--projections-num PROJECTIONS_NUM]
                    [--angle-shift ANGLE_SHIFT]
                    [--angle-coverage ANGLE_COVERAGE]
                    [--output-file OUTPUT_FILE]

Generator of 2D projections of 3D Cryo-Em volumes Args that start with '--'
(eg. --input-file) can also be set in a config file (protein.config or
specified via --config-file). Config file syntax allows: key=value, flag=true,
stuff=[a,b,c] (for details, see syntax at https://goo.gl/R74nmi). If an arg is
specified in more than one place, then commandline values override config file
values which override defaults.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -conf CONFIG_FILE
                        Config file path
  --input-file INPUT_FILE, -in INPUT_FILE
                        Input file of 3D volume (*.mrc format)
  --projections-num PROJECTIONS_NUM, -num PROJECTIONS_NUM
                        Number of 2D projections. Default 5000
  --angle-shift ANGLE_SHIFT, -shift ANGLE_SHIFT
                        Get the start Euler angles that will rotate around
                        axes Z, Y, Z repsectively
  --angle-coverage ANGLE_COVERAGE, -cov ANGLE_COVERAGE
                        The range (size of the interval) of the Euler angles
                        aroung Z, Y, Z axes respectively
  --output-file OUTPUT_FILE, -out OUTPUT_FILE
                        Name of output file containing projections with angles
                        (with the extension)

```

Main use is:
```
# read the settings from config file
python generator.py -config protein.config

# almost half sphere (overrides config default values)
python generator.py -conf protein.config --input-file data/5j0n.mrc -num 5000 -shift 0.0 -shift 0.0 -shift 0.0 -cov 2.0 -cov 0.4 -cov 2.0
```

## Misc information

### Package versions
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

## Notebooks
Notebooks are divided in several phases of development:
- [Phase 0](notebooks/0-preparation): preparation of the simulated data, generating 3D protein's set of 2D projection images and their corresponding angles
- [Phase 1](notebooks/1-phase1): angle recovery using the perfect distances
- [Phase 2](notebooks/2-phase2): distance estimation and angle recovery
- [Phase 3](notebooks/3-phase3): reconstruction of 3D protein structure from 2D projection images and estimated angles (from Phase 1 or Phase 2) 

### Colab Notebooks
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JelenaBanjac/protein-reconstruction/blob/master/notebooks/2-phase2/distance_estimation_and_angle_recovery-test5j0nhalf-cov-polynomialAR.ipynb) Distance Estimation with Angle Recovery Polynomial

## Team
**Student:**  
[Jelena Banjac](https://jelenabanjac.com), jelena.banjac@epfl.ch, Data Science Master Student

**Supervisors:**  
[Laurène Donati](https://people.epfl.ch/laurene.donati?lang=en), laurene.donati@epfl.ch, BIG, EPFL  
[Michaël Defferrard](https://deff.ch/), michael.defferrard@epfl.ch, LTS2, EPFL

**Professor:**  
[Michaël Unser](http://bigwww.epfl.ch/unser/), michael.unser@epfl.ch, BIG, EPFL

