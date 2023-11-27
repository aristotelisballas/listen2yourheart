# Welcome to Listen2YourHeart

> This is the official code for the "A Comprehensive Evaluation of Augmentations 
> for Robust Self-Supervised Contrastive Phonocardiogram Representation Learning"
> paper.

*If not obvious, name inspiration explained [here](https://www.youtube.com/watch?v=yCC_b5WHLX0&themeRefresh=1).*

Listen2YourHeart is an IntelliJ project. The root folder (the one where this file is)
contains two main directories. The `scripts` dir holds all necessary scripts 
for a) downloading all data used in this project (`scripts\datasets`) and b) the 
source code for submitting pretraining and fine-tuning jobs to a 
[SLURM](https://slurm.schedmd.com/documentation.html) workload manager. 
As it names suggests, the `src` dir contains the source code for everything mentioned
in our paper.


## Project Structure

* `scripts`: directory containing scripts for downloading datasets and submitting SLURM jobs.
* `src`: directory containing source code of project
* `requirements.txt`: a simple requirements file that is used to re-create the environment
  of the project.
* `generate-requirements.sh`: convenience script for generating requirements and 
appropriate package dependecies.
* `pcg-ssl.iml`: the project file for IntelliJ.
* `README.md`: this file.


## Datasets
The datasets used in this project are:
* [FPCGDB](https://physionet.org/content/fpcgdb/1.0.0/): Fetal PCG Database
* [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/): A Simultaneous Electrocardiogram and Phonocardiogram Database
* [PASCAL](http://www.peterjbentley.com/heartchallenge/): Classifying Heart Sounds Challenge
* [PhysioNet2016](https://physionet.org/content/challenge-2016/1.0.0/): Classification of Heart Sound Recordings: The PhysioNet/Computing in 
Cardiology Challenge 2016
* [PhysioNet2022](https://moody-challenge.physionet.org/2022/): Heart Murmur Detection from Phonocardiogram Recordings: The George B. 
Moody PhysioNet Challenge 2022

## Quick Start
### Download data
The first thing to do is download all necessary data.
To dowload each dataset you can run the following scripts and
commands from a terminal:

```console
# EPHNOGRAM
./scripts/download-ephnogram.sh

# FPCGDB
./scripts/download-fpcgdb.sh

# PASCAL
./scripts/download-pascal.sh

# PhysioNet2016
wget -r -N -c -np https://physionet.org/files/challenge-2016/1.0.0/

# PhysioNet2022
wget -r -N -c -np https://physionet.org/files/challenge-2022/1.0.0/
```

### Specify Experiment Configuration
Once you have downloaded the data, the next step is to specify all 
parameters needed for the SSL pretraining and downstream task fine-tuning 
experiments

To do that, edit the configuration file --> `./src/configuration/config.yml`










[//]: # (FINAL THING IS TO MENTION OUR PREVIOUS WORK AND CITATIONS)
This repo is an extension of our initial work "[Listen2YourHeart: A Self-Supervised 
Approach for Detecting Murmur in Heart-Beat Sounds](https://ieeexplore.ieee.org/abstract/document/10081680)".