/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
 * You may use, redistribute, and modify this code for non-commercial purposes only, provided that:
 * 1. You give appropriate credit to the original author
 * 2. You indicate if changes were made
 *
 * This code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * For commercial use, please contact the author for licensing terms.
 * Full license text: https://creativecommons.org/licenses/by-nc/4.0/
 */

#include "depth_uncertainty_planner/base_planner.hpp"

using namespace std::chrono;
using namespace common_math;
using namespace depth_uncertainty_planner;

std::vector<MonotonicSegment3> DuPlanner::get_monotonic_segments(const ThirdOrderSegment* segment) {
  // Compute the times at which the segment changes direction along the z-axis
  std::vector<double> roots = segment->get_depth_switching_points_and_terminals();

  // Does not count start and end time
  size_t root_count = roots.size() - 2;

  std::vector<MonotonicSegment3> monotonic_sections;
  // We don't iterate until root_count + 2 because we need to find pairs of roots
  for (unsigned i = 0; i < root_count + 1; i++) {
    if (roots[i] < segment->get_start_time()) {
      // Skip root if it's before start time
      continue;
    } else if (fabs(roots[i] - roots[i + 1]) < 1e-6) {
      // Skip root because it's a duplicate
      continue;
    } else if (roots[i] >= segment->get_end_time()) {
      // We're done because the roots are in ascending order
      break;
    }
    // Add a section between the current root and the next root after checking
    // that the next root is valid We already know that roots[i+1] is greater
    // than the start time because roots[i] is greater than the start time and
    // roots is sorted
    if (roots[i + 1] <= segment->get_end_time()) {
      monotonic_sections.push_back(
        MonotonicSegment3(segment->get_coeffs(), roots[i], roots[i + 1]));
    } else {
      // We're done because the next section is out of the range
      break;
    }
  }
  std::sort(monotonic_sections.begin(), monotonic_sections.end());
  return monotonic_sections;
}

std::vector<CudaMonotonicSegment3> DuPlanner::get_cuda_monotonic_segments(const CudaThirdOrderSegment* segment) {
  // This function exploits the property described in Section II.B of the
  // RAPPIDS paper

  // Compute the coefficients of \dot{d}_z(t)
  CudaVector3d traj_derivative_coeffs[3];
  segment->get_derivative_coeffs(traj_derivative_coeffs);
  double c[3] = {traj_derivative_coeffs[0].z, traj_derivative_coeffs[1].z,
                 traj_derivative_coeffs[2].z};  // Just shortening the names

  // Compute the times at which the segment changes direction along the z-axis
  double switching_points_and_terminals[6];
  // Does not count start and end time
  uint8_t switching_point_count = segment->get_depth_switching_points_and_terminals(switching_points_and_terminals) - 2;

  std::vector<CudaMonotonicSegment3> monotonic_sections;
  // We don't iterate until root_count + 2 because we need to find pairs of roots
  for (unsigned i = 0; i < switching_point_count + 1; i++) {
    if (switching_points_and_terminals[i] < segment->get_start_time()) {
      // Skip root if it's before start time
      continue;
    } else if (fabs(switching_points_and_terminals[i] - switching_points_and_terminals[i + 1]) < 1e-6) {
      // Skip root because it's a duplicate
      continue;
    } else if (switching_points_and_terminals[i] >= segment->get_end_time()) {
      // We're done because the roots are in ascending order
      break;
    }
    // Add a section between the current root and the next root after checking
    // that the next root is valid We already know that roots[i+1] is greater
    // than the start time because roots[i] is greater than the start time and
    // roots is sorted
    if (switching_points_and_terminals[i + 1] <= segment->get_end_time()) {
      CudaVector3d coeffs[4];
      segment->get_coeffs(coeffs);
      monotonic_sections.push_back(
        CudaMonotonicSegment3(coeffs, switching_points_and_terminals[i], switching_points_and_terminals[i + 1]));
    } else {
      // We're done because the next section is out of the range
      break;
    }
  }
  std::sort(monotonic_sections.begin(), monotonic_sections.end());
  return monotonic_sections;
}

bool DuPlanner::is_cuda_segment3_collision_free(const CudaThirdOrderSegment* segment) {
  CudaPinholeCamera cuda_camera = CudaConverter::toCuda(_camera);
  // Split segment into sections with monotonically changing depth
  std::vector<CudaMonotonicSegment3> monotonic_sections =
    get_cuda_monotonic_segments(segment);
  while (monotonic_sections.size() > 0) {
    // Check if we've used up all of our computation time
    if (duration_cast<microseconds>(high_resolution_clock::now() - start_time)
          .count() > int(_allocated_computation_time * 1e6)) {
      return false;
    }

    // Get a monotonic section to check
    CudaMonotonicSegment3 mono_traj = monotonic_sections.back();
    monotonic_sections.pop_back();

    // Find the pixel corresponding to the endpoint of this section (deepest
    // depth)
    CudaVector3d start_point, end_point;
    if (mono_traj.is_increasing_depth()) {
      start_point = mono_traj.get_point(mono_traj.get_start_time());
      end_point = mono_traj.get_point(mono_traj.get_end_time());
    } else {
      start_point = mono_traj.get_point(mono_traj.get_end_time());
      end_point = mono_traj.get_point(mono_traj.get_start_time());
    }

    // Ignore the segment section if it's closer than the minimum collision
    // checking distance
    if (start_point.z < cuda_camera.get_minimum_clear_distance() && end_point.z < cuda_camera.get_minimum_clear_distance()) {
      continue;
    }
    // Try to find pyramid that contains end_point
    int16_t end_point_pixel[2];
    cuda_camera.project_point_to_pixel(end_point, end_point_pixel);
    Pyramid collision_check_pyramid;
    bool pyramid_found = find_containing_pyramid(end_point_pixel[0], end_point_pixel[1], end_point.z, collision_check_pyramid);
    if (!pyramid_found) {
      // No pyramids containing end_point were found, try to make a new pyramid
      if (_pyramids.size() >= _max_num_pyramids ||
          _pyramid_gen_time_nanoseconds > _max_pyramid_gen_time * 1e9) {
        // We've already exceeded the maximum number of allowed pyramids or
        // the maximum time allocated for pyramid generation.
        return false;
      }

      high_resolution_clock::time_point start_inflate =
        high_resolution_clock::now();
      bool pyramid_generated = inflate_pyramid(end_point_pixel[0], end_point_pixel[1],
                                               end_point.z, collision_check_pyramid);
      _pyramid_gen_time_nanoseconds +=
        duration_cast<nanoseconds>(high_resolution_clock::now() - start_inflate)
          .count();

      if (pyramid_generated) {
        // Insert the new pyramid into the list of pyramids found so far
        auto index = std::lower_bound(_pyramids.begin(), _pyramids.end(),
                                      collision_check_pyramid.depth);
        _pyramids.insert(index, collision_check_pyramid);
      } else {
        // No pyramid could be formed, so there must be a collision
        return false;
      }
    }

    // Check if/when the segment intersects a lateral face of the given pyramid.
    double collision_time;
    bool collides_with_pyramid =
      find_deepest_collision_time(mono_traj, collision_check_pyramid, collision_time);

    if (collides_with_pyramid) {
      // The segment collides with at least lateral face of the pyramid. Split
      // the segment where it intersects, and add the section outside the
      // pyramid for further collision checking.
      CudaVector3d coeffs[4];
      mono_traj.get_coeffs(coeffs);
      if (mono_traj.is_increasing_depth()) {
        monotonic_sections.push_back(CudaMonotonicSegment3(
          coeffs, mono_traj.get_start_time(), collision_time));
      } else {
        monotonic_sections.push_back(CudaMonotonicSegment3(
          coeffs, collision_time, mono_traj.get_end_time()));
      }
    }
  }
  return true;
}

bool DuPlanner::find_containing_pyramid(int pixel_x, int pixel_y, double depth, Pyramid& out_pyramid) {
  // This function searches _pyramids for those with base planes at deeper
  // depths than end_point.z
  auto first_pyramid_index =
    std::lower_bound(_pyramids.begin(), _pyramids.end(), depth);
  if (first_pyramid_index != _pyramids.end()) {
    // At least one pyramid exists that has a base plane deeper than end_point.z
    for (std::vector<Pyramid>::iterator it = first_pyramid_index;
         it != _pyramids.end(); ++it) {
      // Check whether end_point is inside the pyramid
      // We need to use the _pyramid_search_pixel_buffer offset here because
      // otherwise we'll try to collision check with the pyramid we just exited
      // while checking the previous section
      if ((*it).left_pix_bound + _pyramid_search_pixel_buffer < pixel_x &&
          pixel_x < (*it).right_pix_bound - _pyramid_search_pixel_buffer &&
          (*it).top_pix_bound + _pyramid_search_pixel_buffer < pixel_y &&
          pixel_y < (*it).bottom_pix_bound - _pyramid_search_pixel_buffer) {
        out_pyramid = *it;
        return true;
      }
    }
  }
  return false;
}

bool DuPlanner::find_deepest_collision_time(CudaMonotonicSegment3 mono_traj,
                                            Pyramid pyramid,
                                            double& out_collision_time) {
  // This function exploits the property described in Section II.C of the
  // RAPPIDS paper

  bool collides_with_pyramid = false;
  if (mono_traj.is_increasing_depth()) {
    out_collision_time = mono_traj.get_start_time();
  } else {
    out_collision_time = mono_traj.get_end_time();
  }
  CudaVector3d coeffs[4];
  mono_traj.get_coeffs(coeffs);
  for (Eigen::Vector3d normal : pyramid.plane_normals) {
    // Compute the coefficients of d(t) (distance to the lateral face of the
    // pyramid)
    double c[4] = {0, 0, 0, 0};
    for (int dim = 0; dim < 3; dim++) {
      c[0] += normal[dim] * coeffs[0][dim];  // t^3
      c[1] += normal[dim] * coeffs[1][dim];  // t^2
      c[2] += normal[dim] * coeffs[2][dim];  // t
      c[3] += normal[dim] * coeffs[3][dim];  // constant
    }

    // Find the times at which the trajectory intersects the plane
    std::vector<double> roots;
    size_t root_count;
    // reducing to finding quadratic roots due to root zero isolation
    if (fabs(c[0]) > 1e-6) {
      std::vector<double> coeffs = {c[0], c[1], c[2], c[3]};
      ThirdOrderPolynomial p3(coeffs, mono_traj.get_start_time(), mono_traj.get_end_time());
      root_count = p3.solve_roots(roots);
    } else {
      std::vector<double> coeffs = {c[1], c[2], c[3]};
      SecondOrderPolynomial p2(coeffs, mono_traj.get_start_time(), mono_traj.get_end_time());
      root_count = p2.solve_roots(roots);
    }
    std::sort(roots.begin(), roots.end());
    if (mono_traj.is_increasing_depth()) {
      // Search backward in time (decreasing depth)
      for (int i = root_count - 1; i >= 0; i--) {
        if (roots[i] > mono_traj.get_end_time()) {
          continue;
        } else if (roots[i] > mono_traj.get_start_time()) {
          if (roots[i] > out_collision_time) {
            // This may seem unnecessary because we are searching an ordered
            // list, but this check is needed because we are checking multiple
            // lateral faces of the pyramid for collisions
            out_collision_time = roots[i];
            collides_with_pyramid = true;
            break;
          }
        } else {
          break;
        }
      }
    } else {
      // Search forward in time (decreasing depth)
      for (int i = 0; i < int(root_count); i++) {
        if (roots[i] < mono_traj.get_start_time()) {
          continue;
        } else if (roots[i] < mono_traj.get_end_time()) {
          if (roots[i] < out_collision_time) {
            out_collision_time = roots[i];
            collides_with_pyramid = true;
            break;
          }
        } else {
          break;
        }
      }
    }
  }
  return collides_with_pyramid;
}

bool DuPlanner::inflate_pyramid(const int& x0, const int& y0, const double& minimumDepth,
                                     Pyramid& outPyramid) {
  // This function is briefly described by Section III.A. of the RAPPIDS paper

  // First check if the sample point violates the field of view constraints
  const double _focalLength = _camera.get_focal_length();
  const double _trueVehicleRadius = _camera.get_true_vehicle_radius();
  const double _vehicleRadiusForPlanning = _camera.get_planning_vehicle_radius();
  const double _minCheckingDist = _camera.get_minimum_clear_distance();
  const int imageEdgeOffset = _focalLength * _trueVehicleRadius / _minCheckingDist;
  const int _pyramidSearchPixelBuffer = _pyramid_search_pixel_buffer;
  const int _imageWidth = _camera.get_width();
  const int _imageHeight = _camera.get_height();

  if (x0 <= imageEdgeOffset + _pyramidSearchPixelBuffer + 1
      || x0 > _imageWidth - imageEdgeOffset - _pyramidSearchPixelBuffer - 1
      || y0 <= imageEdgeOffset + _pyramidSearchPixelBuffer + 1
      || y0 > _imageHeight - imageEdgeOffset - _pyramidSearchPixelBuffer - 1) {
    // Sample point could be in collision with something outside the FOV
    return false;
  }

  // The base plane of the pyramid must be deeper than this depth (written in pixel depth units)
  const float minimumPyramidDepth = minimumDepth + _vehicleRadiusForPlanning;

  // This is the minimum "radius" (really the width/height divided by two) of a valid pyramid
  const int initPixSearchRadius = _focalLength * _vehicleRadiusForPlanning
      / minimumPyramidDepth;

  if (2 * initPixSearchRadius
      >= std::min(_imageWidth, _imageHeight) - 2 * imageEdgeOffset) {
    // The minimum size of the pyramid is larger than the maximum pyramid size
    return false;
  }

  // These edges are the edges of the expanded pyramid before it is shrunk to the final size
  int leftEdge, topEdge, rightEdge, bottomEdge;
  if (y0 - initPixSearchRadius < imageEdgeOffset) {
    topEdge = imageEdgeOffset;
    bottomEdge = topEdge + 2 * initPixSearchRadius;
  } else {
    bottomEdge = std::min(_imageHeight - imageEdgeOffset - 1,
                          y0 + initPixSearchRadius);
    topEdge = bottomEdge - 2 * initPixSearchRadius;
  }
  if (x0 - initPixSearchRadius < imageEdgeOffset) {
    leftEdge = imageEdgeOffset;
    rightEdge = leftEdge + 2 * initPixSearchRadius;
  } else {
    rightEdge = std::min(_imageWidth - imageEdgeOffset - 1,
                         x0 + initPixSearchRadius);
    leftEdge = rightEdge - 2 * initPixSearchRadius;
  }

  // We don't look at any pixels closer than this distance (e.g. if the propellers are in the field of view)
  float ignoreDist = _trueVehicleRadius;
  // For reading the depth value stored in a given pixel.
  float pixDist;

  for (int y = topEdge; y < bottomEdge; y++) {
    for (int x = leftEdge; x < rightEdge; x++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist <= minimumPyramidDepth && pixDist > ignoreDist) {
        // We are unable to inflate a rectangle that will meet the minimum size requirements
        return false;
      }
    }
  }

  // Store the minimum depth pixel value of the expanded pyramid. The base plane of the final pyramid will be
  // this value minus the vehicle radius.
  float maxDepthExpandedPyramid = std::numeric_limits<float>::max();

  // We search each edge of the rectangle until we hit a pixel value closer than minimumPyramidDepth
  // This creates a spiral search pattern around the initial sample point
  // Once all four sides of the pyramid hit either the FOV constraint or a pixel closer than
  // minimumPyramidDepth, we will shrink the pyramid based on the vehicle radius.
  bool rightFree = true, topFree = true, leftFree = true, bottomFree = true;
  while (rightFree || topFree || leftFree || bottomFree) {
    if (rightFree) {
      if (rightEdge < _imageWidth - imageEdgeOffset - 1) {
        for (int y = topEdge; y <= bottomEdge; y++) {
          pixDist = _depth_data[y * _imageWidth + rightEdge + 1];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              rightFree = false;
              rightEdge--;  // Negate the ++ after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        rightEdge++;
      } else {
        rightFree = false;
      }
    }
    if (topFree) {
      if (topEdge > imageEdgeOffset) {
        for (int x = leftEdge; x <= rightEdge; x++) {
          pixDist = _depth_data[(topEdge - 1) * _imageWidth + x];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              topFree = false;
              topEdge++;  // Negate the -- after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        topEdge--;
      } else {
        topFree = false;
      }
    }
    if (leftFree) {
      if (leftEdge > imageEdgeOffset) {
        for (int y = topEdge; y <= bottomEdge; y++) {
          pixDist = _depth_data[y * _imageWidth + leftEdge - 1];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              leftFree = false;
              leftEdge++;  // Negate the -- after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        leftEdge--;
      } else {
        leftFree = false;
      }
    }
    if (bottomFree) {
      if (bottomEdge < _imageHeight - imageEdgeOffset - 1) {
        for (int x = leftEdge; x <= rightEdge; x++) {
          pixDist = _depth_data[(bottomEdge + 1) * _imageWidth + x];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              bottomFree = false;
              bottomEdge--;  // Negate the ++ after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        bottomEdge++;
      } else {
        bottomFree = false;
      }
    }
  }

  // Next, shrink the pyramid according to the vehicle radius
  // Number of pixels to shrink final pyramid. Found by searching outside the boundaries of the expanded pyramid.
  // These edges will be the edges of the final pyramid.
  int rightEdgeShrunk = _imageWidth - 1 - imageEdgeOffset;
  int leftEdgeShrunk = imageEdgeOffset;
  int topEdgeShrunk = imageEdgeOffset;
  int bottomEdgeShrunk = _imageHeight - 1 - imageEdgeOffset;
  int numerator = _focalLength * _vehicleRadiusForPlanning;

  // First check the area between each edge and the edge of the image
  // Check right side
  for (int x = rightEdge; x < _imageWidth; x++) {
    for (int y = topEdge; y <= bottomEdge; y++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        // The pixel is farther away than the minimum checking distance
        if (numerator > (x - rightEdgeShrunk) * pixDist) {
          int rightShrinkTemp = x - int(numerator / pixDist);
          if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking from right will make pyramid invalid
            // Can we shrink from top or bottom instead?
            int topShrinkTemp = y + int(numerator / pixDist);
            int bottomShrinkTemp = y - int(numerator / pixDist);
            if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer
                && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the upper edge, so shrink the lower edge
              bottomEdgeShrunk = bottomShrinkTemp;
            } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the lower edge, so shrink the upper edge
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk);
              int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp);
              if (dShrinkLostArea > uShrinkLostArea) {
                // We lose more area shrinking the bottom side, so shrink the top side
                topEdgeShrunk = topShrinkTemp;
              } else {
                // We lose more area shrinking the top side, so shrink the bottom side
                rightEdgeShrunk = bottomShrinkTemp;
              }
            }
          } else {
            rightEdgeShrunk = rightShrinkTemp;
          }
        }
      }
    }
  }
  // Check left side
  for (int x = leftEdge; x >= 0; x--) {
    for (int y = topEdge; y <= bottomEdge; y++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((leftEdgeShrunk - x) * pixDist < numerator) {
          int leftShrinkTemp = x + int(numerator / pixDist);
          if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking from left will make pyramid invalid
            // Can we shrink from top or bottom instead?
            int topShrinkTemp = y + int(numerator / pixDist);
            int bottomShrinkTemp = y - int(numerator / pixDist);
            if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer
                && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the upper edge, so shrink the lower edge
              bottomEdgeShrunk = bottomShrinkTemp;
            } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the lower edge, so shrink the upper edge
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk);
              int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp);
              if (dShrinkLostArea > uShrinkLostArea) {
                // We lose more area shrinking the bottom side, so shrink the top side
                topEdgeShrunk = topShrinkTemp;
              } else {
                // We lose more area shrinking the top side, so shrink the bottom side
                bottomEdgeShrunk = bottomShrinkTemp;
              }
            }
          } else {
            leftEdgeShrunk = leftShrinkTemp;
          }
        }
      }
    }
  }
  if (leftEdgeShrunk + _pyramidSearchPixelBuffer
      > rightEdgeShrunk - _pyramidSearchPixelBuffer) {
    // We shrunk the left and right sides so much that the pyramid is too small!
    return false;
  }

  // Check top side
  for (int y = topEdge; y >= 0; y--) {
    for (int x = leftEdge; x <= rightEdge; x++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((topEdgeShrunk - y) * pixDist < numerator) {
          int topShrinkTemp = y + int(numerator / pixDist);
          if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking from top will make pyramid invalid
            // Can we shrink from left or right instead?
            int rightShrinkTemp = x - int(numerator / pixDist);
            int leftShrinkTemp = x + int(numerator / pixDist);
            if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
                && x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the upper right, so shrink the left edge
              leftEdgeShrunk = leftShrinkTemp;
            } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the left edge, so shrink the right edge
              rightEdgeShrunk = rightShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp);
              int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk);
              if (rShrinkLostArea > lShrinkLostArea) {
                // We lose more area shrinking the right side, so shrink the left side
                leftEdgeShrunk = leftShrinkTemp;
              } else {
                // We lose more area shrinking the left side, so shrink the right side
                rightEdgeShrunk = rightShrinkTemp;
              }
            }
          } else {
            topEdgeShrunk = topShrinkTemp;
          }
        }
      }
    }
  }
  // Check bottom side
  for (int y = bottomEdge; y < _imageHeight; y++) {
    for (int x = leftEdge; x <= rightEdge; x++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        // The pixel is farther away than the minimum checking distance
        if (numerator > (y - bottomEdgeShrunk) * pixDist) {
          int bottomShrinkTemp = y - int(numerator / pixDist);
          if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking from top will make pyramid invalid
            // Can we shrink from left or right instead?
            int rightShrinkTemp = x - int(numerator / pixDist);
            int leftShrinkTemp = x + int(numerator / pixDist);
            if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
                && x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the upper right, so shrink the left edge
              leftEdgeShrunk = leftShrinkTemp;
            } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the left edge, so shrink the right edge
              rightEdgeShrunk = rightShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp);
              int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk);
              if (rShrinkLostArea > lShrinkLostArea) {
                // We lose more area shrinking the right side, so shrink the left side
                leftEdgeShrunk = leftShrinkTemp;
              } else {
                // We lose more area shrinking the left side, so shrink the right side
                rightEdgeShrunk = rightShrinkTemp;
              }
            }
          } else {
            bottomEdgeShrunk = bottomShrinkTemp;
          }
        }
      }
    }
  }
  if (topEdgeShrunk + _pyramidSearchPixelBuffer
      > bottomEdgeShrunk - _pyramidSearchPixelBuffer) {
    // We shrunk the top and bottom sides so much that the pyramid has no volume!
    return false;
  }

  // Next, check the corners that we ignored before
  // Check top right corner
  for (int y = topEdge; y >= 0; y--) {
    for (int x = rightEdge; x < _imageWidth; x++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if (numerator > (x - rightEdgeShrunk) * pixDist
            && (topEdgeShrunk - y) * pixDist < numerator) {
          // Both right and top edges could shrink
          int rightShrinkTemp = x - int(numerator / pixDist);
          int topShrinkTemp = y + int(numerator / pixDist);
          if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
              && y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking right edge makes pyramid exclude the starting point, so shrink the top edge
            topEdgeShrunk = topShrinkTemp;
          } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking top edge makes pyramid exclude the starting point, so shrink the right edge
            rightEdgeShrunk = rightShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (rShrinkLostArea > uShrinkLostArea) {
              // We lose more area shrinking the right side, so shrink the top side
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We lose more area shrinking the top side, so shrink the right side
              rightEdgeShrunk = rightShrinkTemp;
            }
          }
        }
      }
    }
  }
  // Check bottom right corner
  for (int y = bottomEdge; y < _imageHeight; y++) {
    for (int x = rightEdge; x < _imageWidth; x++) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if (numerator > (x - rightEdgeShrunk) * pixDist
            && numerator > (y - bottomEdgeShrunk) * pixDist) {
          // Both right and bottom edges could shrink
          int rightShrinkTemp = x - int(numerator / pixDist);
          int bottomShrinkTemp = y - int(numerator / pixDist);
          if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
              && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking right edge makes pyramid exclude the starting point, so shrink the bottom edge
            bottomEdgeShrunk = bottomShrinkTemp;
          } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking bottom edge makes pyramid exclude the starting point, so shrink the right edge
            rightEdgeShrunk = rightShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (rShrinkLostArea > dShrinkLostArea) {
              // We lose more area shrinking the right side, so shrink the bottom side
              bottomEdgeShrunk = bottomShrinkTemp;
            } else {
              // We lose more area shrinking the bottom side, so shrink the right side
              rightEdgeShrunk = rightShrinkTemp;
            }
          }
        }
      }
    }
  }
  // Check top left corner
  for (int y = topEdge; y >= 0; y--) {
    for (int x = leftEdge; x >= 0; x--) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((leftEdgeShrunk - x) * pixDist < numerator
            && (topEdgeShrunk - y) * pixDist < numerator) {
          // Both left and top edges could shrink
          int leftShrinkTemp = x + int(numerator / pixDist);
          int topShrinkTemp = y + int(numerator / pixDist);
          if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer
              && y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking left edge makes pyramid exclude the starting point, so shrink the top edge
            topEdgeShrunk = topShrinkTemp;
          } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking top edge makes pyramid exclude the starting point, so shrink the left edge
            leftEdgeShrunk = leftShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (lShrinkLostArea > uShrinkLostArea) {
              // We lose more area shrinking the left side, so shrink the top side
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We lose more area shrinking the top side, so shrink the left side
              leftEdgeShrunk = leftShrinkTemp;
            }
          }
        }
      }
    }
  }
  // Check bottom left corner
  for (int y = bottomEdge; y < _imageHeight; y++) {
    for (int x = leftEdge; x >= 0; x--) {
      pixDist = _depth_data[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((leftEdgeShrunk - x) * pixDist < numerator
            && numerator > (y - bottomEdgeShrunk) * pixDist) {
          // Both left and bottom edges could shrink
          int leftShrinkTemp = x + int(numerator / pixDist);
          int bottomShrinkTemp = y - int(numerator / pixDist);
          if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer
              && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking left edge makes pyramid exclude the starting point, so shrink the bottom edge
            bottomEdgeShrunk = bottomShrinkTemp;
          } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking bottom edge makes pyramid exclude the starting point, so shrink the left edge
            leftEdgeShrunk = leftShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (lShrinkLostArea > dShrinkLostArea) {
              // We lose more area shrinking the left side, so shrink the bottom side
              bottomEdgeShrunk = bottomShrinkTemp;
            } else {
              // We lose more area shrinking the bottom side, so shrink the left side
              leftEdgeShrunk = leftShrinkTemp;
            }
          }
        }
      }
    }
  }

  int edgesFinal[4] = { rightEdgeShrunk, topEdgeShrunk, leftEdgeShrunk,
      bottomEdgeShrunk };
  double depth = maxDepthExpandedPyramid - _vehicleRadiusForPlanning;

  // Create a new pyramid
  Eigen::Vector3d corners[4];
  // Top right
  corners[0] = _camera.deproject_pixel_to_point(double(edgesFinal[0]), double(edgesFinal[1]), depth);
  // Top left
  corners[1] = _camera.deproject_pixel_to_point(double(edgesFinal[2]), double(edgesFinal[1]), depth);
  // Bottom left
  corners[2] = _camera.deproject_pixel_to_point(double(edgesFinal[2]), double(edgesFinal[3]), depth);
  // Bottom right
  corners[3] = _camera.deproject_pixel_to_point(double(edgesFinal[0]), double(edgesFinal[3]), depth);
  outPyramid = Pyramid(depth, edgesFinal, corners);

  return true;
}
