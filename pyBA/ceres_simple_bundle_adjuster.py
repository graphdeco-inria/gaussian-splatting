""" Contains Ceres Simple Bundle Adjustment in Python

"""

import numpy as np
import argparse
import PyCeres

parser = argparse.ArgumentParser(description='Solves a Bundle Adjustment problem')
parser.add_argument('file', help='File from http://grail.cs.washington.edu/projects/bal')
args = parser.parse_args()

if len(sys.argv) == 1:
    sys.exit("No file provided")

file = args.file

bal_problem = PyCeres.BALProblem()

bal_problem.LoadFile(file)

problem = PyCeres.Problem()

observations = bal_problem.observations()
cameras = bal_problem.cameras()
points = bal_problem.points()

numpy_points = np.array(points)
numpy_points = np.reshape(numpy_points, (-1, 3))
numpy_cameras = np.array(cameras)
numpy_cameras = np.reshape(numpy_cameras, (-1, 9))
print(numpy_points[0])

for i in range(0, bal_problem.num_observations()):
    cost_function = PyCeres.CreateSnavelyCostFunction(observations[2 * i + 0], observations[2 * i + 1])
    cam_index = bal_problem.camera_index(i)
    point_index = bal_problem.point_index(i)
    loss = PyCeres.HuberLoss(0.1)
    problem.AddResidualBlock(cost_function, loss, numpy_cameras[cam_index], numpy_points[point_index])

options = PyCeres.SolverOptions()
options.linear_solver_type = PyCeres.LinearSolverType.DENSE_SCHUR
options.minimizer_progress_to_stdout = True

summary = PyCeres.Summary()
PyCeres.Solve(options, problem, summary)
print(summary.FullReport())

# Compare with CPP version

print(" Running C++ version now ")
PyCeres.SolveBALProblemWithCPP(bal_problem)
cpp_points = bal_problem.points()
cpp_points = np.array(cpp_points)
cpp_points = np.reshape(cpp_points, (-1, 3))
print(" For point 1 Python has a value of " + str(numpy_points[0]) + " \n")
print(" Cpp solved for point 1 a value of " + str(cpp_points[0]))
