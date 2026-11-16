"""Load configuration without mutating caller-owned values."""
import json
from collections.abc import Mapping
from pathlib import Path


def load_config(config):
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    with path.open(encoding='utf-8') as stream:
        if path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError('YAML configuration requires PyYAML; install it in the chosen environment') from exc
            values = yaml.safe_load(stream)
        elif path.suffix.lower() == '.json':
            values = json.load(stream)
        else:
            raise ValueError(f'Unsupported configuration format: {path.suffix}')
    if not isinstance(values, Mapping):
        raise ValueError('configuration must be a mapping')
    return dict(values)
