/*!
 * Rectangular Pyramid Partitioning using Integrated Depth Sensors
 *
 * Copyright 2020 by Junseok Lee <junseok_lee@berkeley.edu>
 * and Nathan Bucki <nathan_bucki@berkeley.edu>
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

#include <unistd.h>
#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include "depth_uncertainty_planner/base_planner.hpp"

// Ruckig
#include <ruckig/ruckig.hpp>
#include <geometry_msgs/TransformStamped.h>

using namespace depth_uncertainty_planner;
using namespace autopilot;
// using namespace RapidQuadrocopterTrajectoryGenerator;
// using namespace RectangularPyramidPlanner;

enum testType {
  CONSERVATIVENESS = 0,
  COLLISION_CHECKING_TIME = 1,
  TRAJECTORY_COVERAGE = 2,
};

typedef struct {
  int w;
  int h;
  double depth_scale;
  double f;
  double cx;
  double cy;
} camera_intrinsics_t;

typedef struct {
  double physical_vehicle_radius;
  double vehicle_radius_for_planning;
  double minimum_collision_distance;
} planner_specs_t;

typedef struct {
  uint num_scenes;             // [1]
  uint num_obstacles;          // [1]
  double obstacles_width_min;  // [m]
  double obstacles_width_max;  // [m]
  double obstacles_depth_min;  // [m]
  double obstacles_depth_max;  // [m]
  uint random_seed;
} synthesis_specs_t;

typedef struct {
  bool display;
  double display_range_min;  // [m]
  double display_range_max;  // [m]
  bool png_output;
  double vx_min;
  double vx_max;
  double vy_min;
  double vy_max;
  double vz_min;
  double vz_max;
  double ay_min;
  double ay_max;
  uint random_seed;
  int test_type;
} benchmark_options_t;

typedef struct {
  std::vector<int> max_num_pyramids;
  int num_traj;
} conservative_test_options_t;

typedef struct {
  double pyramid_gen_time;
  int num_traj;
} collision_time_options_t;

typedef struct {
  double min_comp_time;
  double max_comp_time;
  int num_comp_times;
} traj_coverage_options_t;

typedef struct {
  int count;
  double min;
  double max;
  double median;
  double mean;
} benchmark_stats_t;

template<class T>
static benchmark_stats_t calculate_stats(std::vector<T>& q) {
  benchmark_stats_t stats;
  const auto median_it = q.begin() + q.size() / 2;
  std::nth_element(q.begin(), median_it, q.end());  // nth_element is a partial sorting algorithm that rearranges elements.

  stats.count = q.size();
  stats.min = *std::min_element(q.begin(), q.end());
  stats.max = *std::max_element(q.begin(), q.end());
  stats.mean = std::accumulate(std::begin(q), std::end(q), 0.0) / q.size();  // ! 0.0 double 0.0f float 0 integer
  stats.median = *median_it;

  return stats;
}

static void print_stats(std::ostream& os, const std::string& stats_name,
                        benchmark_stats_t& stats) {
  os << std::left << std::endl << std::setw(60) << "Samples:" << stats.count
     << std::endl << std::setw(60) << stats_name + ".min: " << stats.min
     << std::endl << std::setw(60) << stats_name + ".max: " << stats.max
     << std::endl << std::setw(60) << stats_name + ".mean: " << stats.mean
     << std::endl << std::setw(60) << stats_name + ".median: " << stats.median
     << std::endl;
}

static void display_image(const cv::Mat& depth_image, double pixelMin,
                          double pixelMax) {
  cv::Mat displayable_image;
  double alpha = 255.0 / (pixelMax - pixelMin);
  double beta = -alpha * pixelMin;
  depth_image.convertTo(displayable_image, CV_8UC1, alpha, beta);
  // BLUE = FAR
  // RED = CLOSE
  cv::applyColorMap(displayable_image, displayable_image, cv::COLORMAP_HOT);
  cv::imshow("display", displayable_image);
  cv::waitKey(1);
}

// Include center point of your rectangle, size of your rectangle and the degrees of rotation
static void draw_rotated_rectangle(cv::Mat& image, const uint obstacle_depth,
                                   const cv::Point centerPoint,
                                   const cv::Size rectangleSize,
                                   const double rotationDegrees) {
  cv::Scalar color = cv::Scalar(obstacle_depth);

  // Create the rotated rectangle
  cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

  // We take the edges that OpenCV calculated for us
  cv::Point2f vertices2f[4];
  rotatedRectangle.points(vertices2f);

  // Convert them so we can use them in a fillConvexPoly
  cv::Point vertices[4];
  for (int i = 0; i < 4; ++i) {
    vertices[i] = vertices2f[i];
  }

  // Now we can fill the rotated rectangle with our specified color
  cv::fillConvexPoly(image, vertices, 4, color);
}

static void make_image(const camera_intrinsics_t& camera_intrinsics,
                       const synthesis_specs_t& synthesis_specs,
                       std::mt19937& gen, cv::Mat& out_depth_frame) {  // output
  // * Produces random floating-point values i, uniformly distributed on the interval [a, b).
  std::uniform_real_distribution<> random_angle(0, 180);
  std::uniform_real_distribution<> random_w_SI(
      synthesis_specs.obstacles_width_min, synthesis_specs.obstacles_width_max);  // [m]
  std::uniform_real_distribution<> random_depth_SI(
      synthesis_specs.obstacles_depth_min, synthesis_specs.obstacles_depth_max);  // [m]

  // * Produces random integer values i, uniformly distributed on the closed interval [a, b].
  std::uniform_int_distribution<> random_x_px(0, camera_intrinsics.w - 1);

  const uint RECT_HEIGHT = 65535.0f;
  const float DEPTH_MAX = 65535.0f;  // represent infinity in float image

  cv::Mat depth_mat(camera_intrinsics.h, camera_intrinsics.w, CV_32FC1,
                    cv::Scalar(DEPTH_MAX));
  for (int j = 0; j < synthesis_specs.num_obstacles; j++) {
    int center_x = random_x_px(gen);
    int center_y = static_cast<float>(camera_intrinsics.h)
        / static_cast<float>(camera_intrinsics.w) * center_x;

    // pixelValues = depth / depth_scale
    // e.g. 1 meter = 1000 pixel value for depth_scale = 0.001
    float depth_SI = random_depth_SI(gen);
    float depth_px = depth_SI / camera_intrinsics.depth_scale;  // double to int implicit conversion
    draw_rotated_rectangle(
        depth_mat,
        depth_px,
        cv::Point2f(center_x, center_y),
        cv::Size2f(RECT_HEIGHT,
                   camera_intrinsics.f * random_w_SI(gen) / depth_SI),
        random_angle(gen));
  }

  // Output: depth frame
  out_depth_frame = depth_mat;
}

static void run_trajectory_coverage_benchmark(
    const camera_intrinsics_t& camera_intrinsics,
    const planner_specs_t& planner_specs,
    const synthesis_specs_t& synthesis_specs,
    const benchmark_options_t& benchmark_options,
    const traj_coverage_options_t& traj_coverage_options,
    boost::property_tree::ptree& root) {

  if (benchmark_options.display) {
    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);
  }

  std::mt19937 image_rdgen(synthesis_specs.random_seed);
  std::mt19937 state_rdgen(benchmark_options.random_seed);
  std::uniform_real_distribution<> random_vx(benchmark_options.vx_min,
                                             benchmark_options.vx_max);
  std::uniform_real_distribution<> random_vy(benchmark_options.vy_min,
                                             benchmark_options.vy_max);
  std::uniform_real_distribution<> random_vz(benchmark_options.vz_min,
                                             benchmark_options.vz_max);
  std::uniform_real_distribution<> random_ay(benchmark_options.ay_min,
                                             benchmark_options.ay_max);

  boost::property_tree::ptree comp_time_ptree,
                              avg_traj_gen_ptree,
                              avg_traj_check_ptree,
                              avg_collision_free_traj_ptree,
                              best_cost_ptree,
                              avg_traj_gen_db_ptree,
                              avg_traj_check_db_ptree,
                              avg_collision_free_traj_db_ptree,
                              best_cost_db_ptree,
                              avg_traj_sam_to_ptree,
                              avg_traj_check_to_ptree,
                              avg_collision_free_traj_to_ptree,
                              best_cost_to_ptree;
  for (int k = 0; k < traj_coverage_options.num_comp_times; k++) {
    double a = traj_coverage_options.min_comp_time;
    double b = traj_coverage_options.max_comp_time;
    double n = traj_coverage_options.num_comp_times;
    double base = pow(b / a, 1 / (n - 1));  // This increasing computation time geometrically, so we have more resolution at smaller computation times
    double comp_time = a * pow(base, k);  // When k == n-1, compTime = b
    std::vector<int> num_traj_gen, num_traj_gen_db, num_traj_sam_to, num_traj_check, num_traj_check_db, num_traj_check_to, num_collision_free_traj, num_collision_free_traj_db, num_collision_free_traj_to;
    std::vector<double> best_cost, best_cost_db, best_cost_to;
    for (int i = 0; i < synthesis_specs.num_scenes; i++) {
      if (i % 20 == 0) {
        printf("Processed %d frames so far.\n", i);
      }
      cv::Mat depth_mat;
      make_image(camera_intrinsics, synthesis_specs, image_rdgen, depth_mat);

      // ca0: 0.001
      // ca1: 0.0007
      // ca2: 0.005
      // cl0: 0.0
      // cl1: 0.0005
      // cl2: 0.0013
      std::vector<double> depth_uncertainty_coeffs;
      depth_uncertainty_coeffs.push_back(0.001);
      depth_uncertainty_coeffs.push_back(0.0007);
      depth_uncertainty_coeffs.push_back(0.005);
      depth_uncertainty_coeffs.push_back(0.0);
      depth_uncertainty_coeffs.push_back(0.0005);
      depth_uncertainty_coeffs.push_back(0.0013);
      // Store the coefficients in a member variable or use them as needed
      PinholeCamera camera(camera_intrinsics.f, camera_intrinsics.cx,
                           camera_intrinsics.cy, depth_mat.cols, depth_mat.rows,
                           depth_uncertainty_coeffs,
                           planner_specs.physical_vehicle_radius,
                           planner_specs.vehicle_radius_for_planning,
                           planner_specs.minimum_collision_distance);

      Eigen::Vector3d exploration_direction(0, 0, 1);
      geometry_msgs::Point goal_in_world_frame;
      goal_in_world_frame.x = exploration_direction.x();
      goal_in_world_frame.y = exploration_direction.y();
      goal_in_world_frame.z = exploration_direction.z();
      // Pass the projected_goal to the RandomTrajectorySampler
      geometry_msgs::TransformStamped world_to_body, body_to_world;
      RandomTrajectorySampler trajectory_sampler(
        camera, 3.0, 1.5, exploration_direction, 0.3, body_to_world,
        goal_in_world_frame, "world", true, 0.5, false, 1);

      DuPlanner planner1(depth_mat, camera, CollisionCheckingMethod::MIDI, 0.5, 10000, 150, false, 10, 16);
      DuPlanner planner2(depth_mat, camera, CollisionCheckingMethod::MIDI, 0.5, 10000, 150, false, 10, 16);
      DuPlanner planner3(depth_mat, camera, CollisionCheckingMethod::MIDI, 0.5, 10000, 150, false, 10, 16);

      double vx = random_vx(state_rdgen);
      double vy = random_vy(state_rdgen);
      double vz = random_vz(state_rdgen);
      double ay = random_ay(state_rdgen);
    //   RapidTrajectoryGenerator traj(
    //         Vec3(0, 0, 0),
    //         Vec3(vx, vy, vz),
    //         Vec3(0, -ay, 0), Vec3(0, 9.81, 0));

        ruckig::InputParameter<3> initial_state;
        initial_state.current_position = {0.0, 0.0, 0.0};
        initial_state.current_velocity = {vx, vy, vz};
        // initial_state.current_acceleration = {0, -ay, 0};
        initial_state.target_velocity = {0.0, 0.0, 0.0};
        // initial_state.target_acceleration = {0.0, 0.0, 0.0};
        initial_state.max_velocity = {3.0, 1.5, 3.0};
        // initial_state.min_acceleration = {-6.0, -3.0, -6.0};
        initial_state.max_acceleration = {6.0, 3.0, 6.0};
        // initial_state.max_jerk = {67.97, 67.97, 67.97};

        ruckig::Trajectory<3> opt_traj;
        ExplorationCost exploration_cost(exploration_direction, TravelingCost::DIRECTION);
        // uniform
        bool ret = planner1.find_lowest_cost_trajectory(initial_state, opt_traj, trajectory_sampler,
                                                                comp_time, &exploration_cost,
                                                                &ExplorationCost::get_cost_wrapper);
        num_traj_gen.push_back(planner1.get_num_trajectories_generated());
        num_traj_check.push_back(planner1.GetNumCollisionChecks());
        num_collision_free_traj.push_back(0);
        best_cost.push_back(planner1.get_checking_time() / planner1.GetNumCollisionChecks());

        // depth-based
        ret = planner2.find_lowest_cost_trajectory(initial_state, opt_traj, trajectory_sampler,
                                                                comp_time, &exploration_cost,
                                                                &ExplorationCost::get_cost_wrapper);
        num_traj_gen_db.push_back(planner2.get_num_trajectories_generated());
        num_traj_check_db.push_back(planner2.GetNumCollisionChecks());
        num_collision_free_traj_db.push_back(0);
        best_cost_db.push_back(planner2.get_total_generation_time() / planner2.get_num_trajectories_generated());

        // time-optimal
        ret = planner3.find_lowest_cost_trajectory(initial_state, opt_traj, trajectory_sampler,
                                                                comp_time, &exploration_cost,
                                                                &ExplorationCost::get_cost_wrapper);
        num_traj_sam_to.push_back(planner3.get_num_trajectories_generated());
        num_traj_check_to.push_back(planner3.GetNumCollisionChecks());
        num_collision_free_traj_to.push_back(0);
        best_cost_to.push_back(planner3.get_checking_time());

    }

    // uniform
    benchmark_stats_t stats_num_traj_gen = calculate_stats<int>(num_traj_gen);
    benchmark_stats_t stats_num_traj_check = calculate_stats<int>(num_traj_check);
    benchmark_stats_t stats_num_collision_free_traj = calculate_stats<int>(num_collision_free_traj);
    benchmark_stats_t stats_best_cost = calculate_stats<double>(best_cost);
    // depth-based
    benchmark_stats_t stats_num_traj_gen_db = calculate_stats<int>(num_traj_gen_db);
    benchmark_stats_t stats_num_traj_check_db = calculate_stats<int>(num_traj_check_db);
    benchmark_stats_t stats_num_collision_free_traj_db = calculate_stats<int>(num_collision_free_traj_db);
    benchmark_stats_t stats_best_cost_db = calculate_stats<double>(best_cost_db);
    // time-optimal
    benchmark_stats_t stats_num_traj_sam_to = calculate_stats<int>(num_traj_sam_to);
    benchmark_stats_t stats_num_traj_check_to = calculate_stats<int>(num_traj_check_to);
    benchmark_stats_t stats_num_collision_free_traj_to = calculate_stats<int>(num_collision_free_traj_to);
    benchmark_stats_t stats_best_cost_to = calculate_stats<double>(best_cost_to);

    printf("Stats for %f s of computation time:", comp_time);
    print_stats(std::cout, "numTrajGen", stats_num_traj_gen);

    boost::property_tree::ptree comp_time_node;
    comp_time_node.put_value(comp_time);
    comp_time_ptree.push_back(std::make_pair("", comp_time_node));

    // Random sampling output data storaging
    boost::property_tree::ptree avg_traj_gen_node;
    avg_traj_gen_node.put_value(stats_num_traj_gen.mean);
    avg_traj_gen_ptree.push_back(std::make_pair("", avg_traj_gen_node));
    // avg traj checked
    boost::property_tree::ptree avg_traj_check_node;
    avg_traj_check_node.put_value(stats_num_traj_check.mean);
    avg_traj_check_ptree.push_back(std::make_pair("", avg_traj_check_node));

    boost::property_tree::ptree avg_collision_free_traj_node;
    avg_collision_free_traj_node.put_value(stats_num_collision_free_traj.mean);
    avg_collision_free_traj_ptree.push_back(std::make_pair("", avg_collision_free_traj_node));

    boost::property_tree::ptree best_cost_node;
    best_cost_node.put_value(stats_best_cost.mean);
    best_cost_ptree.push_back(std::make_pair("", best_cost_node));

    // Depth-based sampling output data storaging
    boost::property_tree::ptree avg_traj_gen_db_node;
    avg_traj_gen_db_node.put_value(stats_num_traj_gen_db.mean);
    avg_traj_gen_db_ptree.push_back(std::make_pair("", avg_traj_gen_db_node));
    // avg traj checked
    boost::property_tree::ptree avg_traj_check_db_node;
    avg_traj_check_db_node.put_value(stats_num_traj_check_db.mean);
    avg_traj_check_db_ptree.push_back(std::make_pair("", avg_traj_check_db_node));
    boost::property_tree::ptree avg_collision_free_traj_db_node;
    avg_collision_free_traj_db_node.put_value(stats_num_collision_free_traj_db.mean);
    avg_collision_free_traj_db_ptree.push_back(std::make_pair("", avg_collision_free_traj_db_node));

    boost::property_tree::ptree best_cost_db_node;
    best_cost_db_node.put_value(stats_best_cost_db.mean);
    best_cost_db_ptree.push_back(std::make_pair("", best_cost_db_node));

    // Time-optimal output data storaging
    // avg traj sampled
    boost::property_tree::ptree avg_traj_sam_to_node;
    avg_traj_sam_to_node.put_value(stats_num_traj_sam_to.mean);
    avg_traj_sam_to_ptree.push_back(std::make_pair("", avg_traj_sam_to_node));
    // avg traj checked
    boost::property_tree::ptree avg_traj_check_to_node;
    avg_traj_check_to_node.put_value(stats_num_traj_check_to.mean);
    avg_traj_check_to_ptree.push_back(std::make_pair("", avg_traj_check_to_node));

    boost::property_tree::ptree avg_collision_free_traj_to_node;
    avg_collision_free_traj_to_node.put_value(stats_num_collision_free_traj_to.mean);
    avg_collision_free_traj_to_ptree.push_back(std::make_pair("", avg_collision_free_traj_to_node));

    boost::property_tree::ptree best_cost_to_node;
    best_cost_to_node.put_value(stats_best_cost_to.mean);
    best_cost_to_ptree.push_back(std::make_pair("", best_cost_to_node));

    root.put_child("CompTime", comp_time_ptree);  // * put_child replaces the node, instead of adding a node like add_child.
    root.put_child("AvgTrajGen", avg_traj_gen_ptree);
    root.put_child("AvgTrajCheck", avg_traj_check_ptree);
    root.put_child("AvgCollisionFreeTraj", avg_collision_free_traj_ptree);
    root.put_child("BestCost", best_cost_ptree);
    // depth-based
    root.put_child("DbAvgTrajGen", avg_traj_gen_db_ptree);
    root.put_child("DbAvgTrajCheck", avg_traj_check_db_ptree);
    root.put_child("DbAvgCollisionFreeTraj", avg_collision_free_traj_db_ptree);
    root.put_child("DbBestCost", best_cost_db_ptree);
    // time-optimal
    root.put_child("ToAvgTrajSam", avg_traj_sam_to_ptree);
    root.put_child("ToAvgTrajCheck", avg_traj_check_to_ptree);
    root.put_child("ToAvgCollisionFreeTraj", avg_collision_free_traj_to_ptree);
    root.put_child("ToBestCost", best_cost_to_ptree);

    std::ofstream json_out("./data/TimeOptimalYour.json");
    write_json(json_out, root);
    json_out.close();
  }
}

int main(int argc, const char* argv[]) {
  using namespace boost::program_options;
  using namespace boost::property_tree;

  try {
    camera_intrinsics_t camera_intrinsics { };
    planner_specs_t planner_specs { };
    synthesis_specs_t synthesis_specs { };
    benchmark_options_t benchmark_options { };

    traj_coverage_options_t traj_coverage_options { };

    options_description desc { "Options" };
    options_description_easy_init opt = desc.add_options();
    opt("help,h", "Help screen");

    opt("test_type",
        value<int>(&benchmark_options.test_type)->default_value(
            TRAJECTORY_COVERAGE),
        "test to be run (CONSERVATIVENESS = 0, COLLISION_CHECKING_TIME = 1, TRAJECTORY_COVERAGE = 2)");

    opt("image-count,n",
        value<uint>(&synthesis_specs.num_scenes)->default_value(1000),
        "number of scenes for synthetic depth images");
    opt("w", value<int>(&camera_intrinsics.w)->default_value(160),
        "synthesized image width");
    opt("h", value<int>(&camera_intrinsics.h)->default_value(120),
        "synthesized image height");
    opt("depth-scale",
        value<double>(&camera_intrinsics.depth_scale)->default_value(1.0),
        "depth scale");
    opt("f", value<double>(&camera_intrinsics.f)->default_value(386.643 / 4.0),
        "synthesized image focal length");  // This has to be scaled when images scaled down.
    opt("cx", value<double>(&camera_intrinsics.cx)->default_value(160.0 / 2.0),
        "synthesized image cx");
    opt("cy", value<double>(&camera_intrinsics.cy)->default_value(120.0 / 2.0),
        "synthesized image cy");

    opt("vehicleRadius",
        value<double>(&planner_specs.physical_vehicle_radius)->default_value(
            0.26),
        "The true physical radius of the vehicle (ignore depth values closer than this) [meters]");
    opt("planningRadius",
        value<double>(&planner_specs.vehicle_radius_for_planning)->default_value(
            0.46),
        "The radius of the vehicle used for planning [meters]");
    opt("minimumCollisionDistance",
        value<double>(&planner_specs.minimum_collision_distance)->default_value(
            1.0),
        "We assume on obstacle is just outside the field of view this distance away [meters]");

    opt("obs-num",
        value<uint>(&synthesis_specs.num_obstacles)->default_value(2),
        "number of obstacles for synthetic depth images");
    opt("obs-w-min",
        value<double>(&synthesis_specs.obstacles_width_min)->default_value(
            0.30),
        "min width of obstacles in [m]");
    opt("obs-w-max",
        value<double>(&synthesis_specs.obstacles_width_max)->default_value(
            0.30),
        "max width of obstacles in [m]");
    opt("obs-d-min",
        value<double>(&synthesis_specs.obstacles_depth_min)->default_value(1.0),
        "min depth of obstacles in [m]");
    opt("obs-d-max",
        value<double>(&synthesis_specs.obstacles_depth_max)->default_value(2.5),
        "max depth of obstacles in [m]");
    opt("obs-seed", value<uint>(&synthesis_specs.random_seed)->default_value(0),
        "random seed for generating images");

    opt("dispaly,d",
        bool_switch(&benchmark_options.display)->default_value(false),
        "display each frame");
    opt("display-min",
        value<double>(&benchmark_options.display_range_min)->default_value(0.0),
        "min of depth range for display");
    opt("display-max",
        value<double>(&benchmark_options.display_range_max)->default_value(4.0),
        "max of depth range for display");
    opt("png-output,o",
        bool_switch(&benchmark_options.png_output)->default_value(false),
        "output png files");
    opt("vx-min", value<double>(&benchmark_options.vx_min)->default_value(-1),
        "min vx [m/s]");
    opt("vx-max", value<double>(&benchmark_options.vx_max)->default_value(1),
        "max vx [m/s]");
    opt("vy-min", value<double>(&benchmark_options.vy_min)->default_value(-1),
        "min vy [m/s]");
    opt("vy-max", value<double>(&benchmark_options.vy_max)->default_value(1),
        "max vy [m/s]");
    opt("vz-min", value<double>(&benchmark_options.vz_min)->default_value(0),
        "min vz [m/s]");
    opt("vz-max", value<double>(&benchmark_options.vz_max)->default_value(1),
        "max vz [m/s]");
    opt("ay-min", value<double>(&benchmark_options.ay_min)->default_value(-5),
        "min ax [m/s]");
    opt("ay-max", value<double>(&benchmark_options.ay_max)->default_value(5),
        "max ay [m/s]");
    opt("bc-seed",
        value<uint>(&benchmark_options.random_seed)->default_value(0),
        "random seed for initial states");
    opt("minCompTimeForTCTest",
        value<double>(&traj_coverage_options.min_comp_time)->default_value(
            0.01),
        "Minimum computation time allocated to planner (for use with TRAJECTORY_COVERAGE only)");
    opt("maxCompTimeForTCTest",
        value<double>(&traj_coverage_options.max_comp_time)->default_value(
            0.02),
        "Maximum computation time allocated to planner (for use with TRAJECTORY_COVERAGE only)");
    opt("numCompTimesForTCTest",
        value<int>(&traj_coverage_options.num_comp_times)->default_value(3),
        "Number of computation times to test between min and max (for use with TRAJECTORY_COVERAGE only)");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << "\n" << desc << "\n";
      return 0;
    }

    std::cout
        << std::left << std::endl
        << "* please make sure the executable is compiled with optimization."
        << std::endl << "* please make sure the CPU governor is performance."
        << std::endl;

    // std::vector<cv::Mat> qDepthFrames;

    ptree root;

    // * each item will be structured as per dot's
    root.put("synthesis_specs.camera_intrinsics.w", camera_intrinsics.w);
    root.put("synthesis_specs.camera_intrinsics.h", camera_intrinsics.h);
    root.put("synthesis_specs.camera_intrinsics.depth_scale",
             camera_intrinsics.depth_scale);
    root.put("synthesis_specs.camera_intrinsics.f", camera_intrinsics.f);
    root.put("synthesis_specs.camera_intrinsics.cx", camera_intrinsics.cx);
    root.put("synthesis_specs.camera_intrinsics.cy", camera_intrinsics.cy);
    root.put("synthesis_specs.num_scenes", synthesis_specs.num_scenes);
    root.put("synthesis_specs.num_obstacles", synthesis_specs.num_obstacles);
    root.put("synthesis_specs.obstacles_width_min",
             synthesis_specs.obstacles_width_min);
    root.put("synthesis_specs.obstacles_width_max",
             synthesis_specs.obstacles_width_max);
    root.put("synthesis_specs.obstacles_depth_min",
             synthesis_specs.obstacles_depth_min);
    root.put("synthesis_specs.obstacles_depth_max",
             synthesis_specs.obstacles_depth_max);
    root.put("synthesis_specs.random_seed", synthesis_specs.random_seed);

    root.put("benchmark_options.display", benchmark_options.display);
    root.put("benchmark_options.display_range_min",
             benchmark_options.display_range_min);
    root.put("benchmark_options.display_range_max",
             benchmark_options.display_range_max);
    root.put("benchmark_options.png_output", benchmark_options.png_output);
    root.put("benchmark_options.vx_min", benchmark_options.vx_min);
    root.put("benchmark_options.vx_max", benchmark_options.vx_max);
    root.put("benchmark_options.vy_min", benchmark_options.vy_min);
    root.put("benchmark_options.vy_max", benchmark_options.vy_max);
    root.put("benchmark_options.vz_min", benchmark_options.vz_min);
    root.put("benchmark_options.vz_max", benchmark_options.vz_max);
    root.put("benchmark_options.ax_min", benchmark_options.ay_min);
    root.put("benchmark_options.ax_max", benchmark_options.ay_max);
    root.put("benchmark_options.random_seed", benchmark_options.random_seed);

    // Run benchmark and save data
    try {
        root.put("traj_coverage_options.min_comp_time",
                 traj_coverage_options.min_comp_time);
        root.put("traj_coverage_options.max_comp_time",
                 traj_coverage_options.max_comp_time);
        root.put("traj_coverage_options.num_comp_times",
                 traj_coverage_options.num_comp_times);

        run_trajectory_coverage_benchmark(camera_intrinsics, planner_specs,
                                          synthesis_specs, benchmark_options,
                                          traj_coverage_options, root);
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
    }

  } catch (const error& ex) {
    std::cerr << ex.what() << '\n';
  }

  return 0;
}
