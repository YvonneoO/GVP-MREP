#include <ros/ros.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <swarm_exp_msgs/LocalTraj.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <yaw_planner/yaw_planner.h>
#include <gcopter/traj_opt.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <list>
#include <string>

ros::Subscriber traj_sub_, odom_sub_;
ros::Publisher cmd_pub_;
ros::Timer control_timer_;

Trajectory<5> traj_;
YawPlanner yaw_traj_;
std::list<swarm_exp_msgs::LocalTraj> traj_queue_;
std::deque<geometry_msgs::PoseStamped> path_queue_;
geometry_msgs::PoseStamped rotation_target_;

Eigen::Vector3d recover_pt_;
Eigen::Vector3d robot_pos_;
nav_msgs::Odometry latest_odom_;

size_t target_idx_;
double start_t_;
double robot_yaw_;
double integral_a_;
double error_a_last_;
double sample_dt_;
double desired_vel_;
double desired_ang_vel_;
double reach_goal_distance_;
double vel_lower_bound_;
double ang_vel_lower_bound_;
double Kp_a_;
double Ki_a_;
double Kd_a_;
double Kp_d_;
double Ki_d_;
double Kd_d_;
double yaw_goal_tolerance_;
double control_rate_;

int traj_state_;
bool path_active_;
bool rotation_mode_;
bool have_odom_;

geometry_msgs::Twist zero_cmd_;
std::string traj_topic_;
std::string odom_topic_;
std::string cmd_topic_;

geometry_msgs::Quaternion YawToQuat(double yaw) {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();
    return tf2::toMsg(q);
}

double WrapAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void ResetHeadingController() {
    integral_a_ = 0.0;
    error_a_last_ = 0.0;
}

void PublishStop() {
    cmd_pub_.publish(zero_cmd_);
}

void BuildRecoverPath() {
    path_queue_.clear();
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = recover_pt_.x();
    pose.pose.position.y = recover_pt_.y();
    pose.pose.position.z = recover_pt_.z();
    pose.pose.orientation = YawToQuat(robot_yaw_);
    path_queue_.push_back(pose);
    target_idx_ = 0;
    path_active_ = true;
    rotation_mode_ = false;
}

void AppendTrajectoryPoint(double t) {
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = ros::Time::now();
    Eigen::Vector3d p = traj_.getPos(t);
    pose.pose.position.x = p.x();
    pose.pose.position.y = p.y();
    pose.pose.position.z = p.z();

    double yaw, yawd, yawdd;
    yaw_traj_.GetCmd(t, yaw, yawd, yawdd);
    pose.pose.orientation = YawToQuat(yaw);
    path_queue_.push_back(pose);
}

void BuildTrajectoryPath() {
    path_queue_.clear();
    rotation_mode_ = false;
    target_idx_ = 0;
    double total_t = traj_.getTotalDuration();
    if (total_t <= 1e-3) {
        path_active_ = false;
        return;
    }

    double t = std::max(sample_dt_, 0.05);
    for (; t < total_t; t += sample_dt_) {
        AppendTrajectoryPoint(t);
    }
    AppendTrajectoryPoint(total_t);
    path_active_ = !path_queue_.empty();
}

void TrajCallback(const swarm_exp_msgs::LocalTrajConstPtr& traj) {
    traj_queue_.push_back(*traj);
}

bool TryUpdateTraj() {
    if (traj_queue_.empty()) return false;
    double cur_t = ros::WallTime::now().toSec();
    if (traj_queue_.front().start_t > cur_t) return false;

    const auto& msg = traj_queue_.front();
    if (msg.state == 1) {
        traj_state_ = 1;
        recover_pt_(0) = msg.recover_pt.x;
        recover_pt_(1) = msg.recover_pt.y;
        recover_pt_(2) = msg.recover_pt.z;
        BuildRecoverPath();
    } else if (msg.state == 2) {
        traj_state_ = 2;
        start_t_ = msg.start_t;
        traj_.clear();
        Eigen::MatrixXd cM(3, 6);
        int col = 0;
        int t_idx = 0;
        for (size_t i = 0; i < msg.coef_p.size(); ++i, ++col) {
            cM(0, col) = msg.coef_p[i].x;
            cM(1, col) = msg.coef_p[i].y;
            cM(2, col) = msg.coef_p[i].z;
            if (col == 5) {
                traj_.emplace_back(double(msg.t_p[t_idx]), cM);
                col = -1;
                ++t_idx;
            }
        }
        yaw_traj_.A_.resize(msg.coef_yaw.size());
        yaw_traj_.T_.resize(msg.t_yaw.size());
        int yaw_t_idx = 0;
        for (size_t i = 0; i < msg.coef_yaw.size(); ++i) {
            yaw_traj_.A_(i) = double(msg.coef_yaw[i]);
            if ((i + 1) % 6 == 0) {
                yaw_traj_.T_(yaw_t_idx) = double(msg.t_yaw[yaw_t_idx]);
                ++yaw_t_idx;
            }
        }
        double p, v, a;
        yaw_traj_.GetCmd(yaw_traj_.T_.sum(), p, v, a);
        BuildTrajectoryPath();
    } else {
        traj_state_ = -1;
        path_active_ = false;
    }
    traj_queue_.pop_front();
    ResetHeadingController();
    return true;
}

void OdomCallback(const nav_msgs::OdometryConstPtr& odom) {
    latest_odom_ = *odom;
    have_odom_ = true;
    robot_pos_.x() = odom->pose.pose.position.x;
    robot_pos_.y() = odom->pose.pose.position.y;
    robot_pos_.z() = odom->pose.pose.position.z;
    Eigen::Quaterniond q(odom->pose.pose.orientation.w,
                         odom->pose.pose.orientation.x,
                         odom->pose.pose.orientation.y,
                         odom->pose.pose.orientation.z);
    robot_yaw_ = std::atan2(q.toRotationMatrix()(1, 0), q.toRotationMatrix()(0, 0));
}

bool IsWaypointReached(const geometry_msgs::PoseStamped& target) {
    Eigen::Vector2d delta(target.pose.position.x - robot_pos_.x(),
                          target.pose.position.y - robot_pos_.y());
    return delta.norm() <= reach_goal_distance_;
}

bool ShouldAlignYaw(const geometry_msgs::PoseStamped& target) {
    tf2::Quaternion tf_q;
    tf2::fromMsg(target.pose.orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    double yaw_error = WrapAngle(yaw - robot_yaw_);
    return std::abs(yaw_error) > yaw_goal_tolerance_;
}

bool IsYawAligned(const geometry_msgs::PoseStamped& target) {
    tf2::Quaternion tf_q;
    tf2::fromMsg(target.pose.orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    double yaw_error = WrapAngle(yaw - robot_yaw_);
    return std::abs(yaw_error) <= yaw_goal_tolerance_;
}

geometry_msgs::Twist ComputeControl(const geometry_msgs::PoseStamped& target, bool rotation_only) {
    geometry_msgs::Twist cmd;
    Eigen::Vector3d target_global(target.pose.position.x,
                                  target.pose.position.y,
                                  target.pose.position.z);
    Eigen::Quaterniond quat(latest_odom_.pose.pose.orientation.w,
                            latest_odom_.pose.pose.orientation.x,
                            latest_odom_.pose.pose.orientation.y,
                            latest_odom_.pose.pose.orientation.z);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    Eigen::Matrix4d map2body = Eigen::Matrix4d::Identity();
    map2body.block<3, 3>(0, 0) = rot;
    map2body(0, 3) = latest_odom_.pose.pose.position.x;
    map2body(1, 3) = latest_odom_.pose.pose.position.y;
    map2body(2, 3) = latest_odom_.pose.pose.position.z;
    Eigen::Matrix4d body2map = map2body.inverse();
    Eigen::Vector3d target_local = body2map.block<3, 3>(0, 0) * target_global +
                                   body2map.block<3, 1>(0, 3);

    double target_angle = 0.0;
    double current_angle = 0.0;
    if (rotation_only) {
        current_angle = robot_yaw_;
        tf2::Quaternion tf_q;
        tf2::fromMsg(target.pose.orientation, tf_q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        target_angle = yaw;
    } else {
        target_angle = std::atan2(target_local(1), target_local(0));
        if (std::abs(target_local(0)) < 1e-3) {
            target_angle = (target_local(1) >= 0.0) ? M_PI / 2.0 : -M_PI / 2.0;
        }
        if (target_angle > M_PI / 2.0) target_angle -= M_PI;
        else if (target_angle < -M_PI / 2.0) target_angle += M_PI;
    }

    double error_angle = WrapAngle(target_angle - current_angle);
    if (!rotation_only) {
        if (error_angle > M_PI / 2.0) error_angle -= M_PI;
        else if (error_angle < -M_PI / 2.0) error_angle += M_PI;
    }

    integral_a_ += error_angle;
    double derivative = error_angle - error_a_last_;
    error_a_last_ = error_angle;

    double output_angle_vel = Kp_a_ * error_angle + Ki_a_ * integral_a_ + Kd_a_ * derivative;
    if (rotation_only) {
        if (std::abs(output_angle_vel) >= desired_ang_vel_) {
            output_angle_vel = std::copysign(desired_ang_vel_, output_angle_vel);
        } else if (std::abs(output_angle_vel) < ang_vel_lower_bound_) {
            output_angle_vel = std::copysign(ang_vel_lower_bound_, output_angle_vel);
        }
        cmd.angular.z = output_angle_vel;
        cmd.linear.x = 0.0;
        return cmd;
    }

    double error_dis = std::hypot(target_local(0), target_local(1));
    double forward_sign = (target_local(0) >= 0.0) ? 1.0 : -1.0;
    if (std::abs(target_local(0)) < 1e-3) {
        forward_sign = (target_local(1) >= 0.0) ? 1.0 : -1.0;
    }
    double output_linear_vel = 0.0;
    if (error_dis > reach_goal_distance_) {
        output_linear_vel = Kp_d_ * error_dis * forward_sign;
        if (std::abs(output_linear_vel) >= desired_vel_) {
            output_linear_vel = std::copysign(desired_vel_, output_linear_vel);
        } else if (std::abs(output_linear_vel) < vel_lower_bound_) {
            output_linear_vel = std::copysign(vel_lower_bound_, output_linear_vel);
        }
    }

    cmd.angular.z = output_angle_vel;
    cmd.linear.x = output_linear_vel;
    return cmd;
}

void ControlLoop(const ros::TimerEvent&) {
    while (TryUpdateTraj()) {
    }
    if (!have_odom_) return;

    if (rotation_mode_) {
        geometry_msgs::Twist cmd = ComputeControl(rotation_target_, true);
        cmd_pub_.publish(cmd);
        if (IsYawAligned(rotation_target_)) {
            rotation_mode_ = false;
            PublishStop();
        }
        return;
    }

    if (!path_active_ || target_idx_ >= path_queue_.size()) {
        PublishStop();
        return;
    }

    geometry_msgs::PoseStamped target = path_queue_[target_idx_];
    if (IsWaypointReached(target)) {
        if (target_idx_ >= path_queue_.size() - 1) {
            path_active_ = false;
            if (traj_state_ == 2 && ShouldAlignYaw(target)) {
                rotation_mode_ = true;
                rotation_target_ = target;
                ResetHeadingController();
                PublishStop();
                return;
            }
            PublishStop();
            return;
        }
        ++target_idx_;
        return;
    }

    geometry_msgs::Twist cmd = ComputeControl(target, false);
    cmd_pub_.publish(cmd);
}

void LoadParameters(ros::NodeHandle& nh_private) {
    nh_private.param("/traj_topic", traj_topic_, std::string("/trajectory_cmd"));
    nh_private.param("/odom_topic", odom_topic_, std::string("/odom"));
    nh_private.param("/cmd_topic", cmd_topic_, std::string("/cmd_vel"));
    nh_private.param("/waypoint_sample_dt", sample_dt_, 0.3);
    nh_private.param("/desired_velocity", desired_vel_, 0.5);
    nh_private.param("/desired_angular_velocity", desired_ang_vel_, 1.0);
    nh_private.param("/reach_goal_distance", reach_goal_distance_, 0.2);
    nh_private.param("/velocity_lowerbound", vel_lower_bound_, 0.1);
    nh_private.param("/angular_velocity_lowerbound", ang_vel_lower_bound_, 0.1);
    nh_private.param("/Kp_angle", Kp_a_, 1.0);
    nh_private.param("/Ki_angle", Ki_a_, 0.0);
    nh_private.param("/Kd_angle", Kd_a_, 0.0);
    nh_private.param("/Kp_distance", Kp_d_, 1.0);
    nh_private.param("/Ki_distance", Ki_d_, 0.0);
    nh_private.param("/Kd_distance", Kd_d_, 0.0);
    nh_private.param("/yaw_goal_tolerance", yaw_goal_tolerance_, 0.05);
    nh_private.param("/control_rate", control_rate_, 30.0);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ugv_traj_exec");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    LoadParameters(nh_private);

    traj_state_ = -1;
    path_active_ = false;
    rotation_mode_ = false;
    have_odom_ = false;
    target_idx_ = 0;
    integral_a_ = 0.0;
    error_a_last_ = 0.0;
    zero_cmd_.linear.x = 0.0;
    zero_cmd_.angular.z = 0.0;

    traj_sub_ = nh.subscribe(traj_topic_, 10, &TrajCallback);
    odom_sub_ = nh.subscribe(odom_topic_, 10, &OdomCallback);
    cmd_pub_ = nh.advertise<geometry_msgs::Twist>(cmd_topic_, 10);

    double dt = 1.0 / std::max(control_rate_, 1.0);
    control_timer_ = nh.createTimer(ros::Duration(dt), &ControlLoop);

    ros::spin();
    return 0;
}
