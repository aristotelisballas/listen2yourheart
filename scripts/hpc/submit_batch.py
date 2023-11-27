import os
import subprocess
import time

from glob import glob
from pathlib import Path
from typing import List

from absl import app, flags
from absl.flags import FLAGS


flags.DEFINE_string(
    'conf_path', None,
    "The folder holding the config files for submitting experiments.",
    required=True)

FLAGS = flags.FLAGS

"""
This python script is the main method of sending a batch of experiments to the HPC cluster.
Specifically, it takes as input the directory which holds populated experiment 
configurations for each separate hyperparameters and ssl augmentations. 
    
- argument : conf_path --> the directory holding all config.yml files.
"""


def main(args):
    del args
    config_file_list = _get_file_list(Path(FLAGS.conf_path))
    print(f"Found {len(config_file_list)} total config files!")
    for c_file in config_file_list:
        c_file = Path(c_file)
        exp_name = c_file.parts[-1].split(".yml")[0]
        time.sleep(.7)
        print(f"running script for exp: {exp_name} and config: {c_file})")
        subprocess.run(["./scripts/hpc/submit_experiment.sh", f"{exp_name}", f"{c_file}"],
                       capture_output=True)


def _get_file_list(path: Path) -> List[str]:
    file_list = glob(str(path / '*.yml'))
    file_list.sort()

    return file_list


if __name__ == '__main__':
    app.run(main)
