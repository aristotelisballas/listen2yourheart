from pathlib import Path

import yaml


class Configuration:
    def __init__(self, conf_file: Path):
        print(f'Configuration: loading {conf_file}')

        with open(conf_file) as f:
            config = yaml.safe_load(f)

        common = config['common']
        downstream = config['downstream']
        paths = config['paths']
        ssl = config['ssl']

        # Maybe add assertions for data types

        self.config = config
        self.common = common
        self.downstream = downstream
        self.paths = paths
        self.ssl = ssl

