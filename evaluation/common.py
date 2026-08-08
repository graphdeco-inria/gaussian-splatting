# Shared data structures that simplify metrics evaluation from both datasets

class TargetClassInfo:
    """
    Description of one target class in the main project vocabulary and its detector representation

    name is the main name used by the scene and metrics. It is not a dataset name and it is not the detector's name
    name_by_detector is the detector name written in classes.json
    detector_stored_id is the detector mask ID stored in PNG masks. It is the detector model ID shifted by one so zero remains the background ID
    """

    def __init__(self, name, name_by_detector, detector_stored_id):
        """ Store the main name, detector name and stored detector mask ID """
        self.name = name
        self.name_by_detector = name_by_detector
        self.detector_stored_id = detector_stored_id


class SceneData: # Created when loading data in the scene files
    """
    Ground truth and visibility data in the common scene representation

    The arrays related to vertex all use the same order of vertices in the mesh
    - annotated says whether the source dataset provides a semantic annotation
    - visible says if the vertex was observed by any of the selected camera views
    - classes contains the TargetClassInfo classes evaluated by the pipeline
    """

    def __init__(self, dataset, scene, vertices, semantic_labels,
                 annotated, visible, classes):
        """ 
        Store the scene names, vertex arrays and target classes 
        
        - dataset: The dataset name
        - scene: The scene name within the dataset
        - vertices: 3D coordinates of the mesh vertices in the scene
        - semantic_labels: array representing vertices with the local class ID annotation for each vertex
        - annotated: Boolean mask for vertices with one semantic label (maybe not in the target classes) in the source dataset
        - visible: Boolean mask for vertices observed by the selected camera views
        - classes: List of TargetClassInfo objects describing the target classes evaluated by the pipeline
        """
        self.dataset = dataset
        self.scene = scene
        self.vertices = vertices
        self.semantic_labels = semantic_labels
        self.annotated = annotated
        self.visible = visible
        self.classes = classes

    @property # Property: the method can be called as an attribute
    def class_ids(self):
        """ 
        Return the main class name mapped to the local SceneData ID

        This local ID is the index in this scene's classes list. It is neither the detector mask ID nor the source dataset ID
        """
        return {item.name: local_id for local_id, item in enumerate(self.classes)}

    @property
    def evaluation_mask(self):
        """
        Returns a boolean mask for vertices that should be included in evaluation
        
        Consequences of defining the evaluation mask this way:
        - Vertices that are not annotated or not visible are excluded from evaluation.
        - Vertices that are annotated but not in the target classes are included in evaluation, and can cause false positives.
        """

        # Visibility and annotation are independent conditions for evaluation
        return self.annotated & self.visible

    def class_id(self, name):
        """ Return the SceneData local ID assigned to a main class name """
        return self.class_ids[name]


def safe_name(name):
    """ Make a detector name safe for a file or directory name """
    return name.replace(" ", "_")


def ensure_dir(path):
    """ Ensure the existence of a directory and return its path """
    path.mkdir(parents=True, exist_ok=True) # exist_ok allows cached stages to call this repeatedly.
    return path


def target_classes_by_detector(classes):
    """
    Map each detector name to its complete TargetClassInfo record.
    """
    return {item.name_by_detector: item for item in classes}
