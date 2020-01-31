import yaml
import pulsarbat as pb
from pathlib import Path
from baseband import guppi

def get_config(config_file):
    yaml_config = yaml.safe_load(open(config_file, 'r'))
    config = {'source': yaml_config['source'],
              'DM': pb.DispersionMeasure(yaml_config['DM']),
              'ref_freq': config['ref_freq'] * u.MHz,
              'org': config['org']}
    return config


def get_baseband_handle(org, gpu_num=9):
    assert gpu_num in org['gpu_nums']
    folder = Path(org['folder'].format(gpu_num))
    fs = sorted(folder.glob(org['file_pattern']))
    return guppi.open(fs, 'rs')


config = get_config('B1937+21_58245.yml')
fh = get_baseband_handle(config['org'])

