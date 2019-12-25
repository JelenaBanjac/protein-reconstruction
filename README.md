# DL Cryo-EM 

Scripts to generate a huge amount of 2D projections with corresponding angles of 3D volumes.

## Installation
Create the conda environment in which the project will be ran.
```
# create environment
$ conda env create -f environment.yml

# activate environment
$ conda activate protein_reconstruction
```

[Optional] Test if some dependencies are installed:
```
# tensorflow check
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

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
python generator.py -config protein.config -mrc data/5j0n.mrc -num 5000 -shift 0.0,0.1,0.0 -cov 2.0,0.8,1.0
```

## Misc information

### Package versions
The following versions of the packages are installed with Astra-toolbox installation.
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

# Notebooks

Data generation:
- 5j0n_generate_data_5000_0.5sphere.ipynb
- GeneratingData.ipynb
- Graph-50000HalfAngCoverage.ipynb
- Graph-5000FullAngCoverage.ipynb
- Graph-5000HalfAngCoverage.ipynb
- Graph-5000HalfAngCoverage-WithGaussNoise15.ipynb
- Graph-5000HalfAngCoverage-WithGaussNoise2.ipynb

Different angle coverage visualizations:
- angle_variety.ipynb

Euclidean dP and Angle Recovery:
- bgal_optimization_predicted_angle_and_true_angle.ipynb
- 'bgal_optimization_predicted_angle_and_true_projection_(knn_and_slope).ipynb'
- 'bgal_optimization_predicted_angle_and_true_projection_(knn_and_slope)-LR0.001.ipynb'
- 'bgal_optimization_predicted_angle_and_true_projection_(knn&random_and_slope)-Copy1.ipynb'
- 'bgal_optimization_predicted_angle_and_true_projection_(knn&random_and_slope).ipynb'
- 'bgal_optimization_predicted_angle_and_true_projection_(random_and_polyfit)-CONSTRAINED-TODO.ipynb'
- 'bgal_optimization_predicted_angle_and_true_projection_(random_and_polyfit).ipynb'
- 'bgal_optimization_predicted_angle_and_true_projection_(random_and_slope).ipynb'
- optimization_predicted_angle_and_true_projection.ipynb

- 5j0n_optimization_predicted_angle_and_true_projection_0.5angcov-corrected-%.ipynb
- 5j0n_optimization_predicted_angle_and_true_projection_0.5angcov-corrected.ipynb
- 5j0n_optimization_predicted_angle_and_true_projection_0.5angcov.ipynb
- 5j0n_optimization_predicted_angle_and_true_projection_0.5angcov-RANDOM-corrected-%.ipynb
- 5j0n_optimization_predicted_angle_and_true_projection_0.5angcov-RANDOM-corrected.ipynb
- 5j0n_optimization_predicted_angle_and_true_projection_0.5angcov-RANDOM.ipynb

????????
- angle_optimization.ipynb
- angle_optimization_with_GT-notworking.ipynb

Manifold learning (on 5j0n):
- [MDS](notebooks/5j0n_manifold_learning_MDS.ipynb)
- [Spectral Embedding](notebooks/5j0n_manifold_learning_SpectralEmbedding.ipynb)

K-NN Adjacency matrices:
- [projections and angles](notebooks/knn_adjacency_matrices.ipynb)

Siamese NN (on 5j0n):
- [random sampling](notebooks/Siamese_KERAS-protein-random.ipynb)
- [knn projections and random sampling](notebooks/Siamese_KERAS-protein-knn-and-random.ipynb)
- toberemoved 1K epochs [knn projections and random sampling](notebooks/Siamese_KERAS-protein-epochs1000.ipynb)
