// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Class for loading and holding in memory bundle adjustment problems
// from the BAL (Bundle Adjustment in the Large) dataset from the
// University of Washington.
//
// For more details see http://grail.cs.washington.edu/projects/bal/

#ifndef CERES_EXAMPLES_BAL_PROBLEM_H_
#define CERES_EXAMPLES_BAL_PROBLEM_H_

#include <string>

namespace ceres {
namespace examples {

class BALProblem {
 public:
  explicit BALProblem(const std::string& filename, bool use_quaternions);
  ~BALProblem();

  void WriteToFile(const std::string& filename) const;
  void WriteToPLYFile(const std::string& filename) const;

  // Move the "center" of the reconstruction to the origin, where the
  // center is determined by computing the marginal median of the
  // points. The reconstruction is then scaled so that the median
  // absolute deviation of the points measured from the origin is
  // 100.0.
  //
  // The reprojection error of the problem remains the same.
  void Normalize();

  // Perturb the camera pose and the geometry with random normal
  // numbers with corresponding standard deviations.
  void Perturb(const double rotation_sigma,
               const double translation_sigma,
               const double point_sigma);

  // clang-format off
  int camera_block_size()      const { return use_quaternions_ ? 10 : 9; }
  int point_block_size()       const { return 3;                         }
  int num_cameras()            const { return num_cameras_;              }
  int num_points()             const { return num_points_;               }
  int num_observations()       const { return num_observations_;         }
  int num_parameters()         const { return num_parameters_;           }
  const int* point_index()     const { return point_index_;              }
  const int* camera_index()    const { return camera_index_;             }
  const double* observations() const { return observations_;             }
  const double* parameters()   const { return parameters_;               }
  const double* cameras()      const { return parameters_;               }
  double* mutable_cameras()          { return parameters_;               }
  // clang-format on
  double* mutable_points() {
    return parameters_ + camera_block_size() * num_cameras_;
  }

 private:
  void CameraToAngleAxisAndCenter(const double* camera,
                                  double* angle_axis,
                                  double* center) const;

  void AngleAxisAndCenterToCamera(const double* angle_axis,
                                  const double* center,
                                  double* camera) const;
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  bool use_quaternions_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double* parameters_;
};

}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_BAL_PROBLEM_H_
