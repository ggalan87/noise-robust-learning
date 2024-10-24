import os
from pathlib import Path
import time

def compare(base_root, my_root, theirs_root):
    def find_diff(root_path):
        start_file_name = 'config.yaml'
        end_file_keyword = 'events.out.*'

        start_file = root_path / start_file_name
        end_file = next(root_path.glob(end_file_keyword))

        start_time = os.path.getmtime(start_file)
        end_time = os.path.getmtime(end_file)

        return end_time - start_time

    base_time = find_diff(base_root)
    my_time = find_diff(my_root)
    theirs_time = find_diff(theirs_root)

    my_overhead = ((my_time / base_time) * 100) - 100
    theirs_overhead = ((theirs_time / base_time) * 100) - 100
    print(f'my overhead: +{my_overhead}%')
    print(f'my overhead: +{theirs_overhead}%')


def compare_market():
    root = Path('/media/amidemo/Data/object_classifier_data/logs/lightning_logs/market1501_LitSolider')
    base_version = 2
    my_version = 35
    theirs_version = 34

    base_root = root / f'version_{base_version}'
    my_root = root / f'version_{my_version}'
    theirs_root = root / f'version_{theirs_version}'

    compare(base_root, my_root, theirs_root)


def compare_duke():
    root = Path('/media/amidemo/Data/object_classifier_data/logs/lightning_logs/dukemtmcreid_LitSolider')
    base_version = 58
    my_version = 54
    theirs_version = 62

    base_root = root / f'version_{base_version}'
    my_root = root / f'version_{my_version}'
    theirs_root = root / f'version_{theirs_version}'

    compare(base_root, my_root, theirs_root)

compare_market()
compare_duke()