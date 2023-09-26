""" Contains Ceres Simple Bundle Adjustment in Python

"""

import numpy as np
import argparse
import sys
sys.path.insert(0, '/home/xiaoyun/mylibs')
import PyCeres
from jax import grad

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

# f(x,y) = (1-x)^2 + 100(y - x^2)^2;
def CalcCost(x, y):
    return 10000 * (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
class CameraRelativePoseCostFunction(PyCeres.CostFunction):
    def __init__(self):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2])

    def Evaluate(self, parameters, residuals, jacobians):
        x = parameters[0][0]
        y = parameters[0][1]
        residuals[0] = CalcCost(x, y)
        if not (jacobians is None):
            jacobians[0][0], jacobians[0][1] = grad(CalcCost, (0, 1))(x, y)
            # jacobians[0][0] = grad(CalcCost)(x)
            # jacobians[0][1] = grad(CalcCost)(y)
        return True

parameters = [-1.2, 1.0]
np_params = np.array(parameters)

for i in range(0, bal_problem.num_observations()):
    cost_function = PyCeres.CreateSnavelyCostFunction(observations[2 * i + 0], observations[2 * i + 1])
    cam_index = bal_problem.camera_index(i)
    point_index = bal_problem.point_index(i)
    loss = PyCeres.HuberLoss(0.1)
    problem.AddResidualBlock(cost_function, loss, numpy_cameras[cam_index], numpy_points[point_index])

cost_function = CameraRelativePoseCostFunction()
loss = PyCeres.HuberLoss(0.1)
problem.AddResidualBlock(cost_function, loss, np_params)

options = PyCeres.SolverOptions()
options.max_num_iterations = 500
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
print(np_params)
