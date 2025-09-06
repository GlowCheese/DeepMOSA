import os
import pkgutil
import setuptools
import pandas as pd
from pathlib import Path
from pynguin.config import Algorithm
from pynguin.config.main import DeepMosaConfiguration

DEEPMOSA_VERSION = ('v3', 'v4')[
    not DeepMosaConfiguration.async_enabled]

ALGORITHM_MAP = {
    'MOSA': (Algorithm.MOSA, 'm'),
    'CODAMOSA': (Algorithm.CODAMOSA, 'b'),
    'DYNAMOSA': (Algorithm.DYNAMOSA, 'r'),
    'DEEPMOSA': (Algorithm.DEEPMOSA, 'm')
}

BASE_STAT_PATH = 'report/{}/statistics.csv'
EXT_STAT_PATHS = []

if (ext_path:=Path('report-ext')).is_dir():
    for path in ext_path.iterdir():
        base_path = path / '{}' / 'statistics.csv'
        EXT_STAT_PATHS.append(str(base_path))


def find_all_modules(project_path):
    'Find all modules in a project using `setuptools.find_packages`.'
    packages = setuptools.find_packages(project_path)
    all_modules = []
    for package in packages:
        package_path = os.path.join(project_path, package.replace('.', '/'))
        modules = [module for module in pkgutil.iter_modules([package_path])]
        all_modules.extend([''.join(['{}'.format(package), '.', '{}'.format(
            module.name)]) for module in modules if (not module.ispkg)])
    return all_modules


def check_legit(uppercased_algo, module_name):
    """Check if current module already exists in statistics.csv"""
    if (uppercased_algo == 'DEEPMOSA'):
        uppercased_algo += DEEPMOSA_VERSION
    algo = uppercased_algo.lower()

    cnt = 0
    for base_path in (*EXT_STAT_PATHS, BASE_STAT_PATH):
        stat_path = base_path.format(algo)
        if not os.path.isfile(stat_path):
            continue
        df = pd.read_csv(stat_path)
        if (df['TargetModule'] == module_name).any():
            return False
        if base_path == BASE_STAT_PATH:
            while (df['RunId'] == f'{uppercased_algo}_{cnt}').any():
                cnt += 1
    return cnt
