"""Runtime helpers for configuration and dataset path resolution."""

import os


LEGACY_DATASET_ROOT = (
    r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset'
)

DEFAULT_JSON_FILES = {
    'train': 'Aircraft_train.json',
    'val': 'Aircraft_val.json',
    'test': 'Aircraft_test.json',
}

DEFAULT_IMAGE_DIRS = {
    'train': os.path.join('images', 'train'),
    'val': os.path.join('images', 'val'),
    'test': os.path.join('images', 'test'),
}


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_yaml_config(path):
    if not path or not os.path.exists(path):
        return {}

    try:
        import yaml
    except ImportError:
        print(f"PyYAML not installed - ignoring config file: {path}")
        return {}

    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def default_dataset_root():
    return os.getenv('SLIM_DET_DATASET_ROOT', LEGACY_DATASET_ROOT)


def resolve_path(path_value, base_dir=None):
    if path_value is None:
        return None
    if os.path.isabs(path_value) or base_dir is None:
        return path_value
    return os.path.join(base_dir, path_value)


def resolve_dataset_paths(
    dataset_root=None,
    data_config=None,
    split='train',
    json_path=None,
    images_dir=None,
):
    data_config = data_config or {}
    dataset_root = coalesce(dataset_root, data_config.get('dataset_root'), default_dataset_root())

    resolved_json = resolve_path(
        coalesce(json_path, data_config.get(f'{split}_json'), DEFAULT_JSON_FILES[split]),
        dataset_root,
    )
    resolved_images = resolve_path(
        coalesce(images_dir, data_config.get(f'{split}_images'), DEFAULT_IMAGE_DIRS[split]),
        dataset_root,
    )
    return dataset_root, resolved_json, resolved_images


def require_existing_paths(**paths):
    missing = [
        f"{name}='{path}'"
        for name, path in paths.items()
        if path is not None and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
