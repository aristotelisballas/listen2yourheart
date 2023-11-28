import yaml

from absl import app, flags
from absl.flags import FLAGS
from itertools import product
from pathlib import Path

"""
This Python script populates YAML configuration files with different augmentor pairs.
Specifically, it reads: a) a template YAML conf file (./src/configuration/config.yml), b) the below
augmentations from the dictionary, c) a list of different backbone models and d) generates all 
0vs1 and 1vs1 augmentation .yml config files. 
 
"""

augmentations = {
    'cutoff': ['cutofffilter_250_200_2000',
               'cutofffilter_500_450_2000',
               'cutofffilter_750_700_2000',
               'cutofffilter_250_300_2000',
               'cutofffilter_500_550_2000',
               'cutofffilter_750_800_2000'],

    'fliplr': ['fliplr'],

    'flipud': ['flipud'],

    'fliprandom': ['fliprandom_0.3', 'fliprandom_0.5', 'fliprandom_0.7'],

    'randomscaling': ['randomscaling_0.5_2.0', 'randomscaling_1.0_1.5', 'randomscaling_1.5_2.0'],

    'uniformnoise': ['uniformnoise_-0.001_0.001',
                     'uniformnoise_-0.01_0.01',
                     'uniformnoise_-0.1_0.1'],

    'randomresample': ['randomresample_0.1', 'randomresample_0.3', 'randomresample_0.5',
                       'randomresample_0.7', 'randomresample_0.9'],

}

backbones = ['cnn']  


flags.DEFINE_string('export_dir',
                    None,
                    'The export dir for populated .yml configs.')

flags.DEFINE_string('config_file',
                    None,
                    'The template config file from which all cfg files will be populated.')

FLAGS = flags.FLAGS


def main(args):
    del args

    export_dir: Path = Path(FLAGS.export_dir)

    cfg_file: Path = Path(FLAGS.config_file)

    config: dict = _load_yaml(cfg_file)

    pairs_0_1 = _create_0_1(augmentations)
    pairs_1_1 = _create_1_1(augmentations)

    all_pairs = pairs_0_1 + pairs_1_1
    # all_pairs.append([None, None])

    # total_configs = []
    for pair in all_pairs:
        for model in backbones:
            config['ssl']['augmentation1'] = pair[0]
            config['ssl']['augmentation2'] = pair[1]
            config['ssl']['backbone'] = model
            _dump_yaml(config, export_dir / f'{model}_{pair[0]}_{pair[1]}.yml')
            # total_configs.append(config)


def _load_yaml(cfg_file):
    f = open(cfg_file, 'r')
    config = yaml.safe_load(f)
    f.close()
    return config


def _dump_yaml(yaml_dict, filepath: Path):
    # Create export directory if it does not exist
    filepath.parents[0].mkdir(parents=True, exist_ok=True)

    file = open(filepath, 'w')
    yaml.dump(yaml_dict, file)


def _create_combo_augmentations(d):
    configs = []
    for vcomb in product(*d.values()):
        configs.append(list(vcomb))
        # configs.append(dict(zip(d.keys(), vcomb)))

    return configs


def _create_0_1(a: dict) -> list:
    """
    This script takes a dictionary containing
    keys with augmentation types and values with
    the actual augmentations and outputs a list
    of 0vs1 augmentation pairs for creating experiment
    configurations.
    """
    pairs = []
    for key in a.keys():
        for i in a[key]:
            pairs.append([None, i])

    return pairs


def _create_1_1(a: dict) -> list:
    """
    This script takes a dictionary containing
    keys with augmentation types and values with
    the actual augmentations and outputs a list
    of 1vs1 augmentation pairs for creating experiment
    configurations.
    """
    pairs = []
    keys = list(a.keys())

    for i in range(len(keys) - 1):
        for j in range(i+1, len(keys)):
            subdict = dict((k, a[k]) for k in (keys[i], keys[j]) if k in a)
            pairs += _create_combo_augmentations(subdict)

    return pairs


if __name__ == '__main__':
    app.run(main)
