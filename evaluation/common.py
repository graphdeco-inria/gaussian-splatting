"""Common scene records after adding dataset visibility support."""


class TargetClassInfo:
    def __init__(self, name, name_by_detector, detector_stored_id):
        self.name = name
        self.name_by_detector = name_by_detector
        self.detector_stored_id = detector_stored_id


class SceneData:
    """Vertex-aligned labels, visibility and the classes evaluated in a scene."""

    def __init__(self, dataset, scene, vertices, semantic_labels,
                 instance_labels, annotated, visible,
                 instances_seen_by_2D_masks, classes):
        self.dataset = dataset
        self.scene = scene
        self.vertices = vertices
        self.semantic_labels = semantic_labels
        self.instance_labels = instance_labels
        self.annotated = annotated
        self.visible = visible
        self.instances_seen_by_2D_masks = instances_seen_by_2D_masks
        self.classes = classes

    @property
    def class_ids(self):
        return {item.name: index for index, item in enumerate(self.classes)}

    @property
    def evaluation_mask(self):
        return self.annotated & (self.instance_labels >= 0) & self.visible

    def class_id(self, name):
        return self.class_ids[name]


def safe_name(name):
    return name.replace(" ", "_")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def target_classes_by_detector(classes):
    return {item.name_by_detector: item for item in classes}
