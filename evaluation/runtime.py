# Small wrappers for running the project containers

import os
import subprocess
from pathlib import Path


class Runtime:
    """ Run training, fusion scripts and COLMAP through Docker """

    def __init__(self, repo_root, data_root, train_image="tfgivanverdugo/semantic-fusion-gs-train:cuda11.6",
                 fusion_image="tfgivanverdugo/semantic-fusion-fusion:cuda11.6",
                 colmap_image="tfgivanverdugo/semantic-fusion-colmap:3.13.0-cpu"):

        """ Store host roots and the three container image names """
        self.repo_root = Path(repo_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.train_image = train_image
        self.fusion_image = fusion_image
        self.colmap_image = colmap_image

    def _container_path(self, value):
        """ Convert a host path into its mounted container path """

        # Leave relative arguments and strings not related to paths as is
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            # Repository files are mounted as read-only at /repo
            return "/repo/" + path.relative_to(self.repo_root).as_posix()
        except ValueError:
            pass
        try:
            # Dataset and output files are mounted as read-write at /data
            return "/data/" + path.relative_to(self.data_root).as_posix()
        except ValueError:
            return value

    def _docker_command(self, image, gpu, command):
        """
        Build a Docker command from an image and its command arguments

        gpu dds the NVIDIA runtime when enabled.
        command is the sequence of program arguments that runs inside the container.
        """

        # Start a temporary container that is removed after the command exits
        result = ["docker", "run", "--rm"]
        if gpu:
            # Training and rasterization need access to all visible NVIDIA GPUs
            result += ["--gpus", "all"]
        if hasattr(os, "getuid"):
            # Keep files created in mounted directories owned by the host user
            result += ["--user", f"{os.getuid()}:{os.getgid()}"]

        # Set writable cache locations because the repository mount is read-only
        result += [
            "-e", "HOME=/tmp",
            "-e", "MPLCONFIGDIR=/tmp/matplotlib",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e", "QT_QPA_PLATFORM=offscreen",
            "-v", f"{self.repo_root}:/repo:ro",
            "-v", f"{self.data_root}:/data:rw",
            "-w", "/repo",
            image,
        ]

        # Append the program and its arguments after all Docker options
        return result + list(command)

    def _run(self, command):
        """ Run a prepared command and raise errors from failed stages """

        # Print the command so a failed stage can be reproduced manually
        print("Docker command: ", " ".join(str(item) for item in command))
        subprocess.run(command, check=True, text=True, cwd=str(self.repo_root))

    def run_fusion(self, script, arguments):
        """ Run a repository Python script in the fusion container """

        # Convert all mounted host paths before passing the arguments to Docker
        args = [self._container_path(str(item)) for item in arguments]
        command = self._docker_command(self.fusion_image, True, ["python", script] + args)
        self._run(command)

    def run_fusion_module(self, module, arguments):
        """ Run a Python module in the fusion container """

        # Module execution keeps relative imports working inside the repository
        args = [self._container_path(str(item)) for item in arguments]
        command = self._docker_command(self.fusion_image, True, ["python", "-m", module] + args)
        self._run(command)

    def run_train(self, dataset_dir, model_dir, iterations, resolution, data_device="cuda"):
        """
        Run Gaussian training with the selected data and image settings

        resolution is given to the training script as -r. Values such as 1 and 2 select the original or half image resolution.
        """

        # Build the training arguments
        arguments = [
            "-s", str(dataset_dir),
            "-m", str(model_dir),
            "-r", str(resolution),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--checkpoint_iterations", str(iterations),
            "--data_device", data_device,
        ]

        # Only the dataset and model arguments are mounted paths in this list
        mapped = [self._container_path(item) if index in (1, 3) else item for index, item in enumerate(arguments)]
        command = self._docker_command(self.train_image, True, ["python", "train.py"] + mapped)
        self._run(command)

    def run_colmap(self, arguments):
        """ Run COLMAP in the CPU container with the supplied arguments """

        # COLMAP receives all input and output paths through the shared mounts
        args = list(arguments)
        mapped = [self._container_path(str(item)) for item in args]
        command = self._docker_command(self.colmap_image, False, ["colmap"] + mapped)
        self._run(command)
