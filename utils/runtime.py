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


def normalize_class_names(raw_names):
    if raw_names is None:
        return None
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        ordered = sorted(
            raw_names.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
        )
        return [str(name) for _, name in ordered]
    return None


def default_dataset_root():
    return os.getenv('SLIM_DET_DATASET_ROOT', LEGACY_DATASET_ROOT)


def resolve_path(path_value, base_dir=None):
    if path_value is None:
        return None
    if os.path.isabs(path_value) or base_dir is None:
        return path_value
    return os.path.join(base_dir, path_value)


def infer_label_dir(images_dir):
    if images_dir is None:
        return None

    norm_path = os.path.normpath(images_dir)
    drive, tail = os.path.splitdrive(norm_path)
    parts = [part for part in tail.split(os.sep) if part]

    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        return drive + os.sep + os.path.join(*parts)

    parent = os.path.dirname(norm_path)
    split_name = os.path.basename(norm_path)
    return os.path.join(parent, 'labels', split_name)


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


def resolve_detection_paths(
    config_path=None,
    dataset_root=None,
    split='train',
    images_dir=None,
    labels_dir=None,
):
    config = load_yaml_config(config_path) if config_path else {}
    config_root = os.path.dirname(os.path.abspath(config_path)) if config_path else None

    resolved_root = coalesce(dataset_root, config.get('path'), config.get('dataset_root'))
    if resolved_root is not None and not os.path.isabs(resolved_root):
        candidates = []
        if config_root is not None:
            candidates.append(resolve_path(resolved_root, config_root))
        candidates.append(os.path.abspath(resolved_root))
        resolved_root = next((path for path in candidates if os.path.exists(path)), candidates[0])

    split_images = coalesce(images_dir, config.get(split), config.get(f'{split}_images'))
    resolved_images = resolve_path(
        split_images,
        resolved_root or config_root,
    )

    split_labels = coalesce(labels_dir, config.get(f'{split}_labels'))
    resolved_labels = resolve_path(
        split_labels,
        resolved_root or config_root,
    )
    if resolved_labels is None:
        resolved_labels = infer_label_dir(resolved_images)

    class_names = normalize_class_names(config.get('names'))
    num_classes = config.get('nc')
    if num_classes is None and class_names is not None:
        num_classes = len(class_names)

    return {
        'dataset_root': resolved_root or config_root,
        'images_dir': resolved_images,
        'labels_dir': resolved_labels,
        'class_names': class_names,
        'num_classes': num_classes,
    }


def require_existing_paths(**paths):
    missing = [
        f"{name}='{path}'"
        for name, path in paths.items()
        if path is not None and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
