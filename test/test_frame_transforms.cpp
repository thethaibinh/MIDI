/**
 * Unit tests for frame transform and conversion utilities:
 *   - frame_transforms.hpp: FLU↔RDF, RFU↔RDF body/camera transforms
 *   - common.hpp: body_rdf_from_flu, body_flu_from_rdf, quaternion math
 *   - ros_eigen.hpp: toRosPoint, toRosVec3, toEigen conversions
 *   - geometry_eigen_conversions.h: geometryToEigen, eigenToGeometry
 *
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 * Licensed under CC BY-NC 4.0.
 */
#include <gtest/gtest.h>
#include <cmath>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <Eigen/Dense>

#include <common_math/frame_transforms.hpp>
#include <common.hpp>
#include <common_math/ros_eigen.hpp>
#include <common_math/geometry_eigen_conversions.h>

using namespace frame_transform;
using namespace quadrotor_common;

// ============================================================================
// frame_transforms.hpp — FLU (body) ↔ RDF (camera)
// ============================================================================

TEST(FrameTransform, BodyToCamera_ArrayToPoint) {
  // FLU (1, 0, 0) = Forward → camera Z
  std::array<double, 3> body = {1.0, 0.0, 0.0};
  geometry_msgs::msg::Point cam;
  transform_body_to_camera(body, cam);
  EXPECT_NEAR(cam.x, 0.0, 1e-10);
  EXPECT_NEAR(cam.y, 0.0, 1e-10);
  EXPECT_NEAR(cam.z, 1.0, 1e-10);
}

TEST(FrameTransform, BodyToCamera_LeftBecomesRight) {
  // FLU (0, 1, 0) = Left → camera -X (Right is +X, Left is -X in RDF)
  std::array<double, 3> body = {0.0, 1.0, 0.0};
  geometry_msgs::msg::Point cam;
  transform_body_to_camera(body, cam);
  EXPECT_NEAR(cam.x, -1.0, 1e-10);
  EXPECT_NEAR(cam.y, 0.0, 1e-10);
  EXPECT_NEAR(cam.z, 0.0, 1e-10);
}

TEST(FrameTransform, BodyToCamera_UpBecomesDown) {
  // FLU (0, 0, 1) = Up → camera -Y (Down is +Y in RDF)
  std::array<double, 3> body = {0.0, 0.0, 1.0};
  geometry_msgs::msg::Vector3 cam;
  transform_body_to_camera(body, cam);
  EXPECT_NEAR(cam.x, 0.0, 1e-10);
  EXPECT_NEAR(cam.y, -1.0, 1e-10);
  EXPECT_NEAR(cam.z, 0.0, 1e-10);
}

TEST(FrameTransform, CameraToBody_ArrayToPoint) {
  // RDF (0, 0, 1) = Forward → body X (Forward)
  std::array<double, 3> cam = {0.0, 0.0, 1.0};
  geometry_msgs::msg::Point body;
  transform_camera_to_body(cam, body);
  EXPECT_NEAR(body.x, 1.0, 1e-10);
  EXPECT_NEAR(body.y, 0.0, 1e-10);
  EXPECT_NEAR(body.z, 0.0, 1e-10);
}

TEST(FrameTransform, CameraToBody_EigenToPoint) {
  Eigen::Vector3d cam(2.0, -3.0, 5.0);
  geometry_msgs::msg::Point body;
  transform_camera_to_body(cam, body);
  // body.x = cam.z, body.y = -cam.x, body.z = -cam.y
  EXPECT_NEAR(body.x, 5.0, 1e-10);
  EXPECT_NEAR(body.y, -2.0, 1e-10);
  EXPECT_NEAR(body.z, 3.0, 1e-10);
}

TEST(FrameTransform, RoundTripBodyCameraBody) {
  // FLU → RDF → FLU should be identity
  std::array<double, 3> original = {1.5, -2.3, 4.7};
  geometry_msgs::msg::Point cam;
  transform_body_to_camera(original, cam);
  geometry_msgs::msg::Point body;
  transform_camera_to_body(cam, body);
  EXPECT_NEAR(body.x, original[0], 1e-10);
  EXPECT_NEAR(body.y, original[1], 1e-10);
  EXPECT_NEAR(body.z, original[2], 1e-10);
}

TEST(FrameTransform, PointOverloads_Consistency) {
  // Point→Point and Vector3→Vector3 should give same result
  geometry_msgs::msg::Point p;
  p.x = 1.0; p.y = 2.0; p.z = 3.0;
  geometry_msgs::msg::Vector3 v;
  v.x = 1.0; v.y = 2.0; v.z = 3.0;

  geometry_msgs::msg::Point cam_p;
  geometry_msgs::msg::Vector3 cam_v;
  transform_body_to_camera(p, cam_p);
  transform_body_to_camera(v, cam_v);

  EXPECT_NEAR(cam_p.x, cam_v.x, 1e-10);
  EXPECT_NEAR(cam_p.y, cam_v.y, 1e-10);
  EXPECT_NEAR(cam_p.z, cam_v.z, 1e-10);
}

// ============================================================================
// frame_transforms.hpp — RFU ↔ RDF (OmniDrones)
// ============================================================================

TEST(FrameTransform, RFU_BodyToCamera) {
  // RFU (0, 1, 0) = Forward → camera Z (Forward)
  geometry_msgs::msg::Point body;
  body.x = 0.0; body.y = 1.0; body.z = 0.0;
  geometry_msgs::msg::Point cam;
  transform_body_rfu_to_camera(body, cam);
  EXPECT_NEAR(cam.x, 0.0, 1e-10);
  EXPECT_NEAR(cam.y, 0.0, 1e-10);
  EXPECT_NEAR(cam.z, 1.0, 1e-10);
}

TEST(FrameTransform, RFU_UpBecomesNegDown) {
  // RFU (0, 0, 1) = Up → camera Y = -1 (Down is +Y in RDF)
  geometry_msgs::msg::Vector3 body;
  body.x = 0.0; body.y = 0.0; body.z = 1.0;
  geometry_msgs::msg::Vector3 cam;
  transform_body_rfu_to_camera(body, cam);
  EXPECT_NEAR(cam.x, 0.0, 1e-10);
  EXPECT_NEAR(cam.y, -1.0, 1e-10);
  EXPECT_NEAR(cam.z, 0.0, 1e-10);
}

TEST(FrameTransform, RFU_RoundTrip) {
  std::array<double, 3> cam_arr = {1.5, -2.3, 4.7};
  geometry_msgs::msg::Point body;
  transform_camera_to_body_rfu(cam_arr, body);
  geometry_msgs::msg::Point cam_back;
  transform_body_rfu_to_camera(body, cam_back);
  EXPECT_NEAR(cam_back.x, cam_arr[0], 1e-10);
  EXPECT_NEAR(cam_back.y, cam_arr[1], 1e-10);
  EXPECT_NEAR(cam_back.z, cam_arr[2], 1e-10);
}

// ============================================================================
// common.hpp — body_rdf_from_flu / body_flu_from_rdf
// ============================================================================

TEST(CommonTransforms, RdfFromFlu_Eigen) {
  // FLU (F, L, U) → RDF (-L, -U, F)
  Eigen::Vector3d flu(3.0, 4.0, 5.0);
  Eigen::Vector3d rdf = body_rdf_from_flu_eigen(flu);
  EXPECT_NEAR(rdf.x(), -4.0, 1e-10);
  EXPECT_NEAR(rdf.y(), -5.0, 1e-10);
  EXPECT_NEAR(rdf.z(), 3.0, 1e-10);
}

TEST(CommonTransforms, RdfFromFlu_GeometryVector3) {
  geometry_msgs::msg::Vector3 flu;
  flu.x = 1.0; flu.y = 2.0; flu.z = 3.0;
  auto rdf = body_rdf_from_flu_geometry_vector3(flu);
  EXPECT_NEAR(rdf.x, -2.0, 1e-10);
  EXPECT_NEAR(rdf.y, -3.0, 1e-10);
  EXPECT_NEAR(rdf.z, 1.0, 1e-10);
}

TEST(CommonTransforms, RdfFromFlu_GeometryPoint) {
  geometry_msgs::msg::Point flu;
  flu.x = 1.0; flu.y = 2.0; flu.z = 3.0;
  auto rdf = body_rdf_from_flu_geometry_point(flu);
  EXPECT_NEAR(rdf.x, -2.0, 1e-10);
  EXPECT_NEAR(rdf.y, -3.0, 1e-10);
  EXPECT_NEAR(rdf.z, 1.0, 1e-10);
}

TEST(CommonTransforms, FluFromRdf_GeometryVector3) {
  // RDF (R, D, F) → FLU (F, -R, -D)
  Eigen::Vector3d rdf(1.0, 2.0, 3.0);
  auto flu = body_flu_from_rdf_geometry_vector3(rdf);
  EXPECT_NEAR(flu.x, 3.0, 1e-10);
  EXPECT_NEAR(flu.y, -1.0, 1e-10);
  EXPECT_NEAR(flu.z, -2.0, 1e-10);
}

TEST(CommonTransforms, FluFromRdf_GeometryPoint) {
  geometry_msgs::msg::Point rdf;
  rdf.x = 1.0; rdf.y = 2.0; rdf.z = 3.0;
  auto flu = body_flu_from_rdf_geometry_point(rdf);
  EXPECT_NEAR(flu.x, 3.0, 1e-10);
  EXPECT_NEAR(flu.y, -1.0, 1e-10);
  EXPECT_NEAR(flu.z, -2.0, 1e-10);
}

TEST(CommonTransforms, RoundTrip_FluRdfFlu) {
  Eigen::Vector3d original(1.5, -2.3, 4.7);
  Eigen::Vector3d rdf = body_rdf_from_flu_eigen(original);
  auto flu_msg = body_flu_from_rdf_geometry_vector3(rdf);
  EXPECT_NEAR(flu_msg.x, original.x(), 1e-10);
  EXPECT_NEAR(flu_msg.y, original.y(), 1e-10);
  EXPECT_NEAR(flu_msg.z, original.z(), 1e-10);
}

TEST(CommonTransforms, NeuFromEnu) {
  // ENU (E, N, U) → NEU (N, E, U) — swap x and y
  geometry_msgs::msg::Point enu;
  enu.x = 1.0; enu.y = 2.0; enu.z = 3.0;
  auto neu = neu_from_enu_geometry_point(enu);
  EXPECT_NEAR(neu.x, 2.0, 1e-10);
  EXPECT_NEAR(neu.y, 1.0, 1e-10);
  EXPECT_NEAR(neu.z, 3.0, 1e-10);
}

TEST(CommonTransforms, MapEnuFromNwu) {
  // NWU (N, W, U) → ENU: E = -W, N = N, U = U → result: (-y, x, z)
  auto enu = map_enu_from_nwu_double(1.0, 2.0, 3.0);
  EXPECT_NEAR(enu.x, -2.0, 1e-10);
  EXPECT_NEAR(enu.y, 1.0, 1e-10);
  EXPECT_NEAR(enu.z, 3.0, 1e-10);
}

// ============================================================================
// common.hpp — quaternion math
// ============================================================================

TEST(QuaternionMath, IdentityRotationMatrix) {
  // q = (1, 0, 0, 0) → identity matrix
  Eigen::Vector4d q(1, 0, 0, 0);
  Eigen::Matrix3d R = quat2RotMatrix(q);
  EXPECT_TRUE(R.isApprox(Eigen::Matrix3d::Identity(), 1e-10));
}

TEST(QuaternionMath, Rot2QuatIdentity) {
  Eigen::Vector4d q = rot2Quaternion(Eigen::Matrix3d::Identity());
  // Should be (1, 0, 0, 0) or (-1, 0, 0, 0)
  EXPECT_NEAR(std::abs(q(0)), 1.0, 1e-10);
  EXPECT_NEAR(q(1), 0.0, 1e-10);
  EXPECT_NEAR(q(2), 0.0, 1e-10);
  EXPECT_NEAR(q(3), 0.0, 1e-10);
}

TEST(QuaternionMath, Quat2RotRoundTrip) {
  // 90° rotation about Z: q = (cos45, 0, 0, sin45)
  double c = std::cos(M_PI / 4.0);
  double s = std::sin(M_PI / 4.0);
  Eigen::Vector4d q(c, 0, 0, s);
  Eigen::Matrix3d R = quat2RotMatrix(q);

  // R should rotate (1,0,0) to (0,1,0)
  Eigen::Vector3d rotated = R * Eigen::Vector3d(1, 0, 0);
  EXPECT_NEAR(rotated.x(), 0.0, 1e-10);
  EXPECT_NEAR(rotated.y(), 1.0, 1e-10);
  EXPECT_NEAR(rotated.z(), 0.0, 1e-10);

  // Round-trip: rot2Quaternion should recover q (up to sign)
  Eigen::Vector4d q_back = rot2Quaternion(R);
  EXPECT_NEAR(std::abs(q.dot(q_back)), 1.0, 1e-6);
}

TEST(QuaternionMath, QuatMultiplication_Identity) {
  Eigen::Vector4d identity(1, 0, 0, 0);
  Eigen::Vector4d q(0.5, 0.5, 0.5, 0.5);
  Eigen::Vector4d result = quatMultiplication(q, identity);
  EXPECT_NEAR(result(0), q(0), 1e-10);
  EXPECT_NEAR(result(1), q(1), 1e-10);
  EXPECT_NEAR(result(2), q(2), 1e-10);
  EXPECT_NEAR(result(3), q(3), 1e-10);
}

TEST(QuaternionMath, QuatMultiplication_InverseGivesIdentity) {
  // q * q_conjugate = identity (for unit quaternion)
  Eigen::Vector4d q(0.5, 0.5, 0.5, 0.5);  // unit quaternion
  Eigen::Vector4d q_conj(0.5, -0.5, -0.5, -0.5);
  Eigen::Vector4d result = quatMultiplication(q, q_conj);
  EXPECT_NEAR(result(0), 1.0, 1e-10);
  EXPECT_NEAR(result(1), 0.0, 1e-10);
  EXPECT_NEAR(result(2), 0.0, 1e-10);
  EXPECT_NEAR(result(3), 0.0, 1e-10);
}

TEST(QuaternionMath, QuaternionToEuler_Identity) {
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d euler = ::quaternionToEulerAnglesZYX(q);
  EXPECT_NEAR(euler(0), 0.0, 1e-10);  // roll
  EXPECT_NEAR(euler(1), 0.0, 1e-10);  // pitch
  EXPECT_NEAR(euler(2), 0.0, 1e-10);  // yaw
}

TEST(QuaternionMath, QuaternionToEuler_90Yaw) {
  // 90° yaw: quaternion = (cos(45°), 0, 0, sin(45°))
  Eigen::Quaterniond q(
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  Eigen::Vector3d euler = ::quaternionToEulerAnglesZYX(q);
  EXPECT_NEAR(euler(0), 0.0, 1e-6);          // roll
  EXPECT_NEAR(euler(1), 0.0, 1e-6);          // pitch
  EXPECT_NEAR(euler(2), M_PI / 2.0, 1e-6);   // yaw
}

// ============================================================================
// ros_eigen.hpp — toRosPoint / toRosVec3 / toEigen
// ============================================================================

TEST(RosEigen, ToRosPointAndBack) {
  Eigen::Vector3d v(1.5, -2.3, 4.7);
  auto p = toRosPoint(v);
  Eigen::Vector3d back = toEigen(p);
  EXPECT_TRUE(back.isApprox(v, 1e-10));
}

TEST(RosEigen, ToRosVec3AndBack) {
  Eigen::Vector3d v(1.5, -2.3, 4.7);
  auto vec = toRosVec3(v);
  Eigen::Vector3d back = toEigen(vec);
  EXPECT_TRUE(back.isApprox(v, 1e-10));
}

TEST(RosEigen, ToRosQuaternionAndBack) {
  Eigen::Quaterniond q(
    Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 1, 1).normalized()));
  auto ros_q = toRosQuaternion(q);
  Eigen::Quaterniond back = geometryToEigen(ros_q);
  EXPECT_NEAR(q.w(), back.w(), 1e-10);
  EXPECT_NEAR(q.x(), back.x(), 1e-10);
  EXPECT_NEAR(q.y(), back.y(), 1e-10);
  EXPECT_NEAR(q.z(), back.z(), 1e-10);
}

// ============================================================================
// geometry_eigen_conversions.h — geometryToEigen / eigenToGeometry
// ============================================================================

TEST(GeometryEigenConversions, PointRoundTrip) {
  Eigen::Vector3d v(1.0, 2.0, 3.0);
  auto p = eigenToGeometryPoint(v);
  Eigen::Vector3d back = geometryToEigen(p);
  EXPECT_TRUE(back.isApprox(v, 1e-10));
}

TEST(GeometryEigenConversions, Vec3RoundTrip) {
  Eigen::Vector3d v(-1.0, 0.5, 3.14);
  auto vec = eigenToGeometryVec3(v);
  Eigen::Vector3d back = geometryToEigen(vec);
  EXPECT_TRUE(back.isApprox(v, 1e-10));
}

TEST(GeometryEigenConversions, QuaternionRoundTrip) {
  Eigen::Quaterniond q(
    Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitY()));
  auto ros_q = eigenToGeometry(q);
  Eigen::Quaterniond back = geometryToEigen(ros_q);
  // Quaternions equal up to sign
  EXPECT_NEAR(std::abs(q.dot(back)), 1.0, 1e-10);
}

TEST(GeometryEigenConversions, EigenToGeometry_BackwardCompat) {
  // eigenToGeometry(Vector3d) returns Point (backward compat)
  Eigen::Vector3d v(1.0, 2.0, 3.0);
  geometry_msgs::msg::Point p = eigenToGeometry(v);
  EXPECT_NEAR(p.x, v.x(), 1e-10);
  EXPECT_NEAR(p.y, v.y(), 1e-10);
  EXPECT_NEAR(p.z, v.z(), 1e-10);
}
