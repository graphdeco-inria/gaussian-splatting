#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements:
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy
import argparse
import associate
from scipy.spatial.transform import Rotation as R
import json


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(
        numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)
    ).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib.

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = numpy.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(
        description="""
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    """
    )
    parser.add_argument(
        "first_file",
        help="ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "second_file",
        help="estimated trajectory (format: timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "--offset",
        help="time offset added to the timestamps of the second file (default: 0.0)",
        default=0.0,
    )
    parser.add_argument(
        "--scale",
        help="scaling factor for the second trajectory (default: 1.0)",
        default="1.0",
    )
    parser.add_argument(
        "--max_difference",
        help="maximally allowed time difference for matching entries (default: 0.02)",
        default=0.02,
    )
    parser.add_argument(
        "--save",
        help="save aligned second trajectory to disk (format: stamp2 x2 y2 z2)",
    )
    parser.add_argument(
        "--save_associations",
        help="save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)",
    )
    parser.add_argument(
        "--save_gt_json",
        help="save gt trajectory aligned to second trajectory to disk (format: id, [R|t])",
    )
    parser.add_argument(
        "--plot",
        help="plot the first and the aligned second trajectory to an image (format: png)",
        default="plot.png",
    )
    parser.add_argument(
        "--verbose",
        help="print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)",
        action="store_true",
    )
    args = parser.parse_args()

    gt_list = associate.read(args.first_file)
    estimated_list = associate.read(args.second_file)

    matches = associate.associate(
        gt_list, estimated_list, float(args.offset), float(args.max_difference)
    )
    if len(matches) < 2:
        sys.exit(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?"
        )
    print("len(matches)=", len(matches))

    first_xyz = numpy.matrix(
        [[float(value) for value in gt_list[a][0:3]] for a, b in matches]
    ).transpose()
    second_xyz = numpy.matrix(
        [
            [float(value) * eval(args.scale) for value in estimated_list[b][0:3]]
            for a, b in matches
        ]
    ).transpose()
    rot, trans, trans_error = align(second_xyz, first_xyz)

    second_xyz_aligned = rot * second_xyz + trans

    first_stamps = list(gt_list.keys())
    first_stamps.sort()
    first_xyz_full = numpy.matrix(
        [[float(value) for value in gt_list[b][0:3]] for b in first_stamps]
    ).transpose()

    second_stamps = list(estimated_list.keys())
    second_stamps.sort()
    second_xyz_full = numpy.matrix(
        [
            [float(value) * eval(args.scale) for value in estimated_list[b][0:3]]
            for b in second_stamps
        ]
    ).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans

    if args.verbose:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))

        print(
            "absolute_translational_error.rmse %f m"
            % numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
        )
        print("absolute_translational_error.mean %f m" % numpy.mean(trans_error))
        print("absolute_translational_error.median %f m" % numpy.median(trans_error))
        print("absolute_translational_error.std %f m" % numpy.std(trans_error))
        print("absolute_translational_error.min %f m" % numpy.min(trans_error))
        print("absolute_translational_error.max %f m" % numpy.max(trans_error))
    else:
        print(
            "absolute_translational_error.rmse %f m"
            % numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
        )

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write(
            "\n".join(
                [
                    "%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2)
                    for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(
                        matches,
                        first_xyz.transpose().A,
                        second_xyz_aligned.transpose().A,
                    )
                ]
            )
        )
        file.close()

    if args.save:
        file = open(args.save, "w")
        file.write(
            "\n".join(
                [
                    "%f " % stamp + " ".join(["%f" % d for d in line])
                    for stamp, line in zip(
                        second_stamps, second_xyz_full_aligned.transpose().A
                    )
                ]
            )
        )
        file.close()

    if args.save_gt_json:
        Two = numpy.identity(4, dtype=float)
        transforms = []

        for ct, (a, b) in enumerate(matches):
            gt_t = numpy.array(gt_list[a][0:3])
            gt_rot = R.from_quat(gt_list[a][3:]).as_matrix()
            xform_mat = numpy.identity(4, dtype=float)
            xform_mat[:3, :3] = gt_rot
            xform_mat[:3, 3] = gt_t

            if ct == 0:
                # set first frame to origin
                R_inv = numpy.linalg.inv(xform_mat[:3, :3])
                t_inv = -R_inv @ xform_mat[:3, 3]
                Two[:3, :3] = R_inv
                Two[:3, 3] = t_inv

                est_t = numpy.array(estimated_list[b][0:3])
                est_rot = R.from_quat(estimated_list[b][3:]).as_matrix()
                temp_mat = numpy.identity(4, dtype=float)
                temp_mat[:3, :3] = est_rot
                temp_mat[:3, 3] = est_t

                Two = temp_mat @ Two

            # transform to origin coordinate
            xform_mat = Two @ xform_mat
            transforms.append({"id": ct, "xform": xform_mat.tolist()})

        with open(args.save_gt_json, "w") as f:
            json.dump(transforms, f, indent=4)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        from matplotlib.patches import Ellipse

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(
            ax, first_stamps, first_xyz_full.transpose().A, "-", "black", "ground truth"
        )
        plot_traj(
            ax,
            second_stamps,
            second_xyz_full_aligned.transpose().A,
            "-",
            "blue",
            "estimated",
        )

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(
            matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A
        ):
            ax.plot([x1, x2], [y1, y2], "-", color="red", label=label)
            label = ""

        ax.legend()

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(args.plot, dpi=90)
