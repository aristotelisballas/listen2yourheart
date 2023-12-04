from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd
from absl import app, flags
from pandas import DataFrame

flags.DEFINE_string('tmp_path', None, 'The path where the experiment folders are', required=True)
flags.DEFINE_string('prefix', 'experiment', 'The prefix for the experiments')
flags.DEFINE_string('result_path', None, 'Where to save the results file')
FLAGS = flags.FLAGS

metrics = ['experiment', 'murmur-f', 'murmur-acc', 'murmur-wacc', 'out-acc',
           'out-wacc', 'out-cost', 'augment1', 'augment2', 'temperature']


def _good_sort(all_paths: List[str]):
    paths = []
    names = []
    for path in all_paths:
        name = Path(path).name
        if name.startswith(FLAGS.prefix):
            paths.append(path)
            names.append(name)

    n: int = len(FLAGS.prefix)
    ids = [int(p[n:]) for p in names]

    sorted_paths = [p for _, p in sorted(zip(ids, paths))]

    return sorted_paths


def gather_augs(lines: List[str], idxs: List[int]):
    augs: List[str] = []
    for idx in idxs:
        aug = lines[idx].split('=')[1].split('\n')[0]
        augs.append(aug)

    return ', '.join(augs)


def get_results():
    experiments = sorted(glob(FLAGS.tmp_path + '/*'))

    experiments = _good_sort(experiments)

    results = []
    for experiment in experiments:
        experiment = Path(experiment)
        scores_csv = experiment / 'scores.csv'
        scores = pd.read_csv(scores_csv, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 21, 22])
        with open(experiment / 'pretrain' / 'flags.txt') as f:
            lines = f.readlines()
        idx1 = [idx for idx, s in enumerate(lines) if 'augmentation1' in s]
        idx2 = [idx for idx, s in enumerate(lines) if 'augmentation2' in s]
        idx3 = [idx for idx, s in enumerate(lines) if 'temperature' in s][0]
        # aug1 = lines[idx1].split('=')[1].split('\n')[0]
        # aug2 = lines[idx2].split('=')[1].split('\n')[0]
        augs1 = gather_augs(lines, idx1)
        augs2 = gather_augs(lines, idx2)
        temperature = lines[idx3].split('=')[1].split('\n')[0]
        tmp = [
            str(experiment.name),
            scores['F-measure'][0],
            scores['Accuracy'][0],
            scores['Weighted Accuracy'][0],
            scores['Accuracy'][2],
            scores['Weighted Accuracy'][2],
            scores['Cost'][2],
            augs1,
            augs2,
            temperature
        ]
        results.append(tmp)

    result_df = pd.DataFrame(results, columns=metrics)

    if FLAGS.result_path is not None:
        result_path: Path = Path(FLAGS.result_path)
        if result_path.is_dir():
            result_path = result_path / (FLAGS.prefix + '.csv')
        result_df.to_csv(result_path)

    return result_df


def create_result_path() -> Optional[Path]:
    result_path: Path = Path(FLAGS.result_path)

    if result_path is None:
        return None

    if result_path.is_dir():
        return result_path / (FLAGS.prefix + '.csv')

    return result_path


def main(args):
    del args

    results: DataFrame = get_results()
    print(results)


if __name__ == '__main__':
    app.run(main)
