/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.edu>
 *
 * This code is free software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This code is distributed in the hope that it will
 * be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the code.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace common_math {

class Pyramid {
 public:
  //! Default constructor. Not a valid pyramid.
  Pyramid()
    : depth(std::numeric_limits<double>::quiet_NaN()),
      right_pix_bound(std::numeric_limits<int>::quiet_NaN()),
      top_pix_bound(std::numeric_limits<int>::quiet_NaN()),
      left_pix_bound(std::numeric_limits<int>::quiet_NaN()),
      bottom_pix_bound(std::numeric_limits<int>::quiet_NaN()) {}

  //! Creates a new pyramid.
  /*!
   * @param depthIn The depth of the base plane (perpendicular to the z-axis) in
   * meters
   * @param edgesIn The position in pixel coordinates of each lateral face of
   * the pyramid. Because we're using the pinhole camera model, when the pyramid
   * is projected into the image plane, it appears as a rectangle. The entries
   * of the array (in order) are the right, top, left, and bottom edges of the
   * rectangle (where right > left and bottom > top).
   * @param corners The position of the corners of the pyramid as written in the
   * camera-fixed frame. Each corner should have the same depth. [meters]
   */
  Pyramid(double depth_in, int edges_in[4], Eigen::Vector3d corners_in[4])
    : depth(depth_in),
      right_pix_bound(edges_in[0]),
      top_pix_bound(edges_in[1]),
      left_pix_bound(edges_in[2]),
      bottom_pix_bound(edges_in[3]) {
    _corners[0] = corners_in[0];
    _corners[1] = corners_in[1];
    _corners[2] = corners_in[2];
    _corners[3] = corners_in[3];
    plane_normals[0] = corners_in[0].cross(corners_in[1]).normalized();
    plane_normals[1] = corners_in[1].cross(corners_in[2]).normalized();
    plane_normals[2] = corners_in[2].cross(corners_in[3]).normalized();
    plane_normals[3] = corners_in[3].cross(corners_in[0]).normalized();
  }

  Pyramid(double depth_in, Eigen::Vector3d corners_in[4])
    : depth(depth_in) {
    _corners[0] = corners_in[0];
    _corners[1] = corners_in[1];
    _corners[2] = corners_in[2];
    _corners[3] = corners_in[3];
    plane_normals[0] = corners_in[0].cross(corners_in[1]).normalized();
    plane_normals[1] = corners_in[1].cross(corners_in[2]).normalized();
    plane_normals[2] = corners_in[2].cross(corners_in[3]).normalized();
    plane_normals[3] = corners_in[3].cross(corners_in[0]).normalized();
  }
  //! We define this operator so that we can sort pyramid by the depth of their
  //! base planes
  bool operator<(const Pyramid& rhs) const { return depth < rhs.depth; }

  //! We define this operator so that we can search a sorted list of pyramids
  //! and ignore those at a shallower depth that a given sample point.
  bool operator<(const double rhs) const { return depth < rhs; }

  //! Depth of the base plane of the pyramid [meters]
  double depth;
  //! Location of the right lateral face projected into the image, making it a
  //! line located between leftPixBound and the image width
  int right_pix_bound;
  //! Location of the top lateral face projected into the image, making it a
  //! line located between 0 and bottomPixBound
  int top_pix_bound;
  //! Location of the left lateral face projected into the image, making it a
  //! line located between 0 and rightPixBound
  int left_pix_bound;
  //! Location of the bottom lateral face projected into the image, making it a
  //! line located between topPixBound and the image height
  int bottom_pix_bound;
  //! Unit normals of the lateral faces of the pyramid
  Eigen::Vector3d plane_normals[4];
  Eigen::Vector3d _corners[4];
};
}  // namespace common_math
