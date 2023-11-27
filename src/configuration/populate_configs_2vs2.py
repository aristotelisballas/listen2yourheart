import yaml

from absl import app, flags
from absl.flags import FLAGS
from itertools import combinations, product
from pathlib import Path

"""
This Python script populates YAML configuration files with different augmentor pairs.
Specifically, it reads: a) a template YAML conf file (./src/configuration/config.yml), b) the below
augmentations from the dictionary, c) a list of different backbone models and d) generates all 
0vs1 and 1vs1 augmentation .yml config files. 

"""

augmentations = {
    'cutoff': ['cutofffilter_250_300_2000'],

    'fliplr': ['fliplr'],

    'flipud': ['flipud'],

    'fliprandom': ['fliprandom_0.7'],

    'randomscaling': ['randomscaling_1.0_1.5'],

    'uniformnoise': ['uniformnoise_-0.01_0.01'],

    # 'randomresample': ['randomresample_0.1', 'randomresample_0.3', 'randomresample_0.5',
    #                    'randomresample_0.7', 'randomresample_0.9']

    'no_aug': ['None']
}

backbones = ['cnn']  # maybe add 'vgg16'

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

    aug_list = list(augmentations.values())

    aug_list_corr = []
    aug_list_final = []

    for x in aug_list:
        for y in x:
            aug_list_corr.append(y)

    for p in combinations(aug_list_corr, 2):
        o = [x for x in aug_list_corr if x > p[0] and x != p[1]]
        for x in combinations(o, 2):
            aug_list_final.append([list(p), list(x)])

    for pair in aug_list_final:
        if 'None' in pair[0]:
            pair[0].remove('None')
        if 'None' in pair[1]:
            pair[1].remove('None')

        for model in backbones:
            config['ssl']['augmentation1'] = pair[0]
            config['ssl']['augmentation2'] = pair[1]
            config['ssl']['backbone'] = model

            pair0_str = str(pair[0]).replace("[", '').replace("]", '')
            pair0_str = pair0_str.replace("'", '').replace(", ", '_')

            pair1_str = str(pair[1]).replace("[", '').replace("]", '')
            pair1_str = pair1_str.replace("'", '').replace(", ", '_')

            _dump_yaml(config, export_dir / f'{model}_{pair0_str}_{pair1_str}.yml')


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
        for j in range(i + 1, len(keys)):
            subdict = dict((k, a[k]) for k in (keys[i], keys[j]) if k in a)
            pairs += _create_combo_augmentations(subdict)

    return pairs


if __name__ == '__main__':
    app.run(main)
