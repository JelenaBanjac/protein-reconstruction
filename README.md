# DL Cryo-EM 

Scripts to generate a huge amount of 2D projections with corresponding angles of 3D volumes.

## Installation
First, install Astra-Toolbox. Astra-Toolbox should be installed as indicated in their [GitHub repo](https://github.com/astra-toolbox/astra-toolbox).
```
# what I concretelly used (other choices can be found in their repo)
$ conda install -c astra-toolbox astra-toolbox
```

Afterwards, in the root directory of the project, run:
```
pip install -r requirements.txt
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
usage: generator.py [-h] --input INPUT [--proj-num PROJ_NUM]
                    [--ang-coverage ANG_COVERAGE] [--ang-shift ANG_SHIFT]
                    [--output OUTPUT]

Generator of 2D projections of 3D Cryo-Em volumes

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -mrc INPUT
                        Input file of 3D volume (*.mrc format)
  --proj-num PROJ_NUM, -num PROJ_NUM
                        Number of 2D projections. Default 50000
  --ang-coverage ANG_COVERAGE, -cov ANG_COVERAGE
                        Angular coverage (0.5: half-sphere, 1: complete
                        sphere). Default 0.5
  --ang-shift ANG_SHIFT, -shift ANG_SHIFT
                        Start of angular coverage. Default pi/2
  --output OUTPUT, -mat OUTPUT
                        Name of output file containing projections with angles
                        (with the extension)
```

Main use is:
```
python generator.py -mrc generated_data/bgal.mrc 

python generator.py -mrc generated_data/bgal.mrc -cov 1
```

## Misc information

### Package versions
The following versions of the packages are installed with Astra-toolbox installation.
```
python-3.6.8
cudnn-7.1.3
cudatoolkit-8.0
```