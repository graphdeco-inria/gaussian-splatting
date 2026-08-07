# Shared data structures and utilities for evaluation of 3D semantic fusion results

class TargetClassInfo:
    """
    Description of one target class and its detector labels.

    name is the main name used by the scene and metrics.
    name_by_detector is the name written by the detector in classes.json.
    detector_stored_id is the numeric label stored in the mask images by the detector.
    """

    def __init__(self, name, name_by_detector, detector_stored_id):
        """ Store the three names and identifiers used for one target class """
        self.name = name
        self.name_by_detector = name_by_detector
        self.detector_stored_id = detector_stored_id


class SceneData: # Created when loading data in the scene files
    """
    Ground truth and visibility data in the common scene representation

    The arrays related to vertex all use the same mesh-vertex order. 
    - annotated says whether the source dataset provides a semantic annotation
    - visible says if the vertex was observed by the selected camera views
    - instances_seen_by_2D_masks records the instances observed by the generated 2D masks
    - classes contains the target classes evaluated by the pipeline
    """

    def __init__(self, dataset, scene, vertices, semantic_labels,
                 instance_labels, annotated, visible, instances_seen_by_2D_masks,
                 classes):
        """ 
        Store the scene names, vertex-aligned arrays and target classes 
        
        - dataset: The dataset name
        - scene: The scene name within the dataset
        - vertices: 3D coordinates of the mesh vertices in the scene
        - semantic_labels: Local semantic class ID from this Scene, as CLASSES is common between datasets
        - instance_labels: Instance ID from the dataset, -1 for vertices not belonging to any instance
        - annotated: Boolean mask for vertices with a semantic label in the source dataset
        - visible: Boolean mask for vertices observed by the selected camera views
        - instances_seen_by_2D_masks: Set of instance IDs observed by the generated 2D masks
        - classes: List of TargetClassInfo objects describing the target classes evaluated by the pipeline
        """
        self.dataset = dataset
        self.scene = scene
        self.vertices = vertices
        self.semantic_labels = semantic_labels
        self.instance_labels = instance_labels
        self.annotated = annotated
        self.visible = visible
        self.instances_seen_by_2D_masks = instances_seen_by_2D_masks
        self.classes = classes

    @property # Property: the method can be called as an attribute
    def class_ids(self):
        """ 
        Returns the class name to class-index mapping 
        
        Important: this is a local index for SceneData for the classes in the scene, not the detector's stored IDs.
        During the evaluation, there are also indexes specific from each dataset
        """
        return {item.name: index for index, item in enumerate(self.classes)}

    @property
    def evaluation_mask(self):
        """
        Primary visible annotated vertex set

        Main class labels may be -1 for annotated non-target classes.
        Those vertices remain in the mask so predictions on them count as false positives

        Consequences of defining the evaluation mask this way:
        - Vertices that are not annotated or not visible are excluded from evaluation.
        - Vertices that are annotated but not in the target classes are included in evaluation, and can cause false positives.
        - Vertices without a valid instance label are excluded from evaluation (even background has a label, so there are not too many of these).
        """

        # Visibility and annotation are independent conditions for evaluation
        return (self.annotated & (self.instance_labels >= 0) & self.visible)

    def class_id(self, name):
        """ Return the local index assigned to a class name """
        return self.class_ids[name]

    def visible_instance_count(self):
        """Count valid instances in the primary evaluation set."""
        values = self.instance_labels[self.evaluation_mask]
        return len({int(value) for value in values if value >= 0})


def safe_name(name):
    """ Make a detector name safe for use as a file or directory name """
    return name.replace(" ", "_")


def ensure_dir(path):
    """ Ensure the existence of a directory and return its path """
    path.mkdir(parents=True, exist_ok=True) # exist_ok allows cached stages to call this repeatedly.
    return path


def target_classes_by_detector(classes):
    """
    Assigns the name to the whole data of the target class
    """
    return {item.name_by_detector: item for item in classes}
