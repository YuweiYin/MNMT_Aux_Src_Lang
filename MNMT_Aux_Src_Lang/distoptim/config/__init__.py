"""distoptim.config package"""
import os
import yaml
import argparse
import logging

LOG = logging.getLogger(__name__)

__version__ = "0.1.1"


def build_hit_config(args):
    args.use_hit = False

    if args.distributed_init_method.startswith('hit'):
        args.device_id = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
        hit_config_file = args.distributed_init_method[3:]
        print(hit_config_file)
        if hit_config_file:
            with open(hit_config_file) as f:
                config = yaml.safe_load(f)
                if config['type'] == 'hit':
                    args.use_hit = True
                print(config)
                return config
        else:
            raise RuntimeError('Missing hit_config_file')
    else:
        config = {'type': 'default'}
        return config
