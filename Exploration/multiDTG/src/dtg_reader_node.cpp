#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Eigen>
#include <queue>
#include <list>
#include <algorithm>

namespace DTG {

struct ReaderNode {
    uint32_t id;
    Eigen::Vector3d pos;
    bool is_f;
    int state; // for H nodes
    int vp_id; // for F nodes
    
    // Dijkstra search state
    double g = std::numeric_limits<double>::infinity();
    uint32_t parent_id = 0;
    bool visited = false;

    struct Edge {
        uint32_t neighbor_id;
        double length_s;
        double length;
        int flag;
        bool is_hf; // true for HF, false for HH
        std::vector<Eigen::Vector3d> path;
    };
    std::vector<Edge> edges;
};

class DTGReader {
public:
    DTGReader(ros::NodeHandle& nh) : nh_(nh) {
        nh.param<std::string>("dtg_log_file", dtg_log_file_, "");
        
        if (dtg_log_file_.empty()) {
            ROS_ERROR("[DTGReader] dtg_log_file parameter is empty!");
        } else {
            ROS_INFO("[DTGReader] Loading CSV file: %s", dtg_log_file_.c_str());
            loadCSV(dtg_log_file_);
        }

        nav_goal_sub_ = nh_.subscribe("/move_base_simple/goal", 10, &DTGReader::navGoalCB, this);
        graph_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("dtg_reader_graph", 10);
        path_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("dtg_reader_path", 10);
        
        vis_timer_ = nh_.createWallTimer(ros::WallDuration(1.0), &DTGReader::visCB, this);

        ROS_INFO("[DTGReader] Initialized. Click points in RViz to find paths.");
    }

private:
    void loadCSV(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            ROS_ERROR("[DTGReader] Failed to open file: %s", filename.c_str());
            return;
        }

        std::string line;
        std::string section = "";

        auto safe_stoi = [](const std::string& s, int default_val = 0) {
            try {
                if (s.empty() || s == " ") return default_val;
                return std::stoi(s);
            } catch (...) { return default_val; }
        };

        auto safe_stoul = [](const std::string& s, uint32_t default_val = 0) {
            try {
                if (s.empty() || s == " ") return default_val;
                return static_cast<uint32_t>(std::stoul(s));
            } catch (...) { return default_val; }
        };

        auto safe_stod = [](const std::string& s, double default_val = 0.0) {
            try {
                if (s.empty() || s == " ") return default_val;
                return std::stod(s);
            } catch (...) { return default_val; }
        };

        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.find("H_NODES") != std::string::npos) section = "H_NODES";
                else if (line.find("F_NODES") != std::string::npos) section = "F_NODES";
                else if (line.find("HH_EDGES") != std::string::npos) section = "HH_EDGES";
                else if (line.find("HF_EDGES") != std::string::npos) section = "HF_EDGES";
                continue;
            }

            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> row;
            while (std::getline(ss, item, ',')) {
                row.push_back(item);
            }

            try {
                if (section == "H_NODES") {
                    if (row.size() < 5) continue;
                    uint32_t id = safe_stoul(row[0]);
                    ReaderNode node;
                    node.id = id;
                    node.pos = Eigen::Vector3d(safe_stod(row[1]), safe_stod(row[2]), safe_stod(row[3]));
                    node.state = safe_stoi(row[4]);
                    node.is_f = false;
                    nodes_[id] = node;
                    h_node_ids_.push_back(id);
                } else if (section == "F_NODES") {
                    if (row.size() < 5) continue;
                    uint32_t id = safe_stoul(row[0]);
                    ReaderNode node;
                    node.id = id;
                    node.pos = Eigen::Vector3d(safe_stod(row[1]), safe_stod(row[2]), safe_stod(row[3]));
                    node.vp_id = safe_stoi(row[4]);
                    node.is_f = true;
                    nodes_[id] = node;
                    f_node_ids_.push_back(id);
                } else if (section == "HH_EDGES") {
                    if (row.size() < 6) continue;
                    uint32_t head = safe_stoul(row[0]);
                    uint32_t tail = safe_stoul(row[1]);
                    double length_s = safe_stod(row[2]);
                    double length = safe_stod(row[3]);
                    int flag = safe_stoi(row[4]);
                    
                    std::vector<Eigen::Vector3d> path;
                    if (row.size() >= 7 && !row[6].empty()) {
                        std::stringstream ss_path(row[6]);
                        std::string pt_str;
                        std::vector<double> coords;
                        while (std::getline(ss_path, pt_str, ';')) {
                            if (pt_str.empty()) continue;
                            coords.push_back(safe_stod(pt_str));
                            if (coords.size() == 3) {
                                path.push_back(Eigen::Vector3d(coords[0], coords[1], coords[2]));
                                coords.clear();
                            }
                        }
                    }

                    if (nodes_.count(head)) {
                        nodes_[head].edges.push_back({tail, length_s, length, flag, false, path});
                    }
                    if (nodes_.count(tail)) {
                        std::vector<Eigen::Vector3d> rev_path = path;
                        std::reverse(rev_path.begin(), rev_path.end());
                        nodes_[tail].edges.push_back({head, length_s, length, flag, false, rev_path});
                    }
                } else if (section == "HF_EDGES") {
                    if (row.size() < 5) continue;
                    uint32_t head = safe_stoul(row[0]);
                    uint32_t tail = safe_stoul(row[1]);
                    double length = safe_stod(row[2]);
                    int flag = safe_stoi(row[3]);
                    
                    std::vector<Eigen::Vector3d> path;
                    if (row.size() >= 6 && !row[5].empty()) {
                        std::stringstream ss_path(row[5]);
                        std::string pt_str;
                        std::vector<double> coords;
                        while (std::getline(ss_path, pt_str, ';')) {
                            if (pt_str.empty()) continue;
                            coords.push_back(safe_stod(pt_str));
                            if (coords.size() == 3) {
                                path.push_back(Eigen::Vector3d(coords[0], coords[1], coords[2]));
                                coords.clear();
                            }
                        }
                    }

                    if (nodes_.count(head)) {
                        // For HF edges, length_s is not applicable, use length for both
                        nodes_[head].edges.push_back({tail, length, length, flag, true, path});
                    }
                }
            } catch (const std::exception& e) {
                ROS_WARN("[DTGReader] Skipping malformed line in section %s: %s (Error: %s)", section.c_str(), line.c_str(), e.what());
            }
        }
        ifs.close();
        ROS_INFO("[DTGReader] Loaded CSV. H-Nodes: %zu, F-Nodes: %zu", h_node_ids_.size(), f_node_ids_.size());
    }

    void navGoalCB(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        Eigen::Vector3d clicked_p(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        
        if (points_.size() >= 2) points_.clear();
        points_.push_back(clicked_p);

        if (points_.size() == 1) {
            ROS_INFO("[DTGReader] Start point set: (%.2f, %.2f, %.2f). Click another point for goal.", 
                     clicked_p.x(), clicked_p.y(), clicked_p.z());
        } else if (points_.size() == 2) {
            ROS_INFO("[DTGReader] Goal point set: (%.2f, %.2f, %.2f).", 
                     clicked_p.x(), clicked_p.y(), clicked_p.z());
            findAndVisualizePath(points_[0], points_[1]);
        }
    }

    void findAndVisualizePath(const Eigen::Vector3d& start, const Eigen::Vector3d& goal) {
        uint32_t startNodeId = findNearestNode(start, false);
        uint32_t goalNodeId = findNearestNode(goal, true);

        if (startNodeId == 0 || goalNodeId == 0) {
            ROS_WARN("[DTGReader] Could not find suitable nearest nodes.");
            return;
        }

        const auto& sn = nodes_[startNodeId];
        const auto& gn = nodes_[goalNodeId];
        ROS_INFO("[DTGReader] Found Nearest Nodes:");
        ROS_INFO("  - Start: ID %u, Pos (%.2f, %.2f, %.2f), Dist to Clicked: %.2f", 
                 startNodeId, sn.pos.x(), sn.pos.y(), sn.pos.z(), (sn.pos - start).norm());
        ROS_INFO("  - Goal:  ID %u, Pos (%.2f, %.2f, %.2f), Dist to Clicked: %.2f", 
                 goalNodeId, gn.pos.x(), gn.pos.y(), gn.pos.z(), (gn.pos - goal).norm());
        
        ROS_INFO("[DTGReader] Finding topological path...");
        for (auto& pair : nodes_) {
            pair.second.g = std::numeric_limits<double>::infinity();
            pair.second.parent_id = 0;
            pair.second.visited = false;
        }

        auto cmp = [](const std::pair<double, uint32_t>& a, const std::pair<double, uint32_t>& b) {
            return a.first > b.first;
        };
        std::priority_queue<std::pair<double, uint32_t>, std::vector<std::pair<double, uint32_t>>, decltype(cmp)> pq(cmp);

        nodes_[startNodeId].g = 0;
        pq.push({0.0, startNodeId});

        bool found = false;
        while (!pq.empty()) {
            uint32_t u = pq.top().second;
            pq.pop();

            if (nodes_[u].visited) continue;
            nodes_[u].visited = true;

            if (u == goalNodeId) {
                found = true;
                break;
            }

            for (const auto& edge : nodes_[u].edges) {
                if (nodes_.count(edge.neighbor_id)) {
                    // Replicate MultiDTG logic for using dangerous (length) or safe (length_s) length
                    double edge_len = edge.length_s;
                    if (edge.flag & 16) edge_len = edge.length;

                    double new_g = nodes_[u].g + edge_len;
                    if (new_g < nodes_[edge.neighbor_id].g) {
                        nodes_[edge.neighbor_id].g = new_g;
                        nodes_[edge.neighbor_id].parent_id = u;
                        pq.push({new_g, edge.neighbor_id});
                    }
                }
            }
        }

        if (found) {
            std::vector<Eigen::Vector3d> full_path;
            uint32_t curr = goalNodeId;
            while (curr != 0) {
                uint32_t prev = nodes_[curr].parent_id;
                if (prev != 0) {
                    // Find edge from prev to curr
                    for (const auto& edge : nodes_[prev].edges) {
                        if (edge.neighbor_id == curr) {
                            for (auto it = edge.path.rbegin(); it != edge.path.rend(); ++it) {
                                full_path.insert(full_path.begin(), *it);
                            }
                            break;
                        }
                    }
                }
                curr = prev;
            }
            publishPathMarkers(full_path);
            ROS_INFO("[DTGReader] Path found! Points: %zu", full_path.size());
        } else {
            ROS_WARN("[DTGReader] No path found.");
        }
    }

    uint32_t findNearestNode(const Eigen::Vector3d& p, bool prefer_f) {
        uint32_t nearestId = 0;
        double minDist = std::numeric_limits<double>::infinity();
        // ROS_INFO("[DTGReader] Finding nearest node to point (%.2f, %.2f, %.2f), nodes size: %zu, prefer_f: %d", p.x(), p.y(), p.z(), nodes_.size(), prefer_f);
        for (const auto& pair : nodes_) {
            if (pair.second.is_f != prefer_f) continue;
            double dist = (pair.second.pos - p).norm();
            // ROS_INFO("[DTGReader] Node ID: %u, Distance: %.2f", pair.first, dist);
            if (dist < minDist) {
                // ROS_INFO("[DTGReader] New nearest node found: ID %u, Distance: %.2f", pair.first, dist);
                minDist = dist;
                nearestId = pair.first;
            }
        }
        return nearestId;
    }

    void publishPathMarkers(const std::vector<Eigen::Vector3d>& path) {
        visualization_msgs::MarkerArray ma;
        visualization_msgs::Marker line;
        line.header.frame_id = "world";
        line.header.stamp = ros::Time::now();
        line.ns = "path";
        line.id = 0;
        line.type = visualization_msgs::Marker::LINE_STRIP;
        line.action = visualization_msgs::Marker::ADD;
        line.scale.x = 0.1;
        line.color.r = 0.0; line.color.g = 1.0; line.color.b = 0.0; line.color.a = 1.0;
        line.pose.orientation.w = 1.0;

        for (const auto& p : path) {
            geometry_msgs::Point ros_p;
            ros_p.x = p.x(); ros_p.y = p.y(); ros_p.z = p.z();
            line.points.push_back(ros_p);
        }
        ma.markers.push_back(line);
        path_pub_.publish(ma);
    }

    void visCB(const ros::WallTimerEvent&) {
        visualization_msgs::MarkerArray ma;
        
        // H-Nodes
        visualization_msgs::Marker h_markers;
        h_markers.header.frame_id = "world";
        h_markers.header.stamp = ros::Time::now();
        h_markers.ns = "h_nodes";
        h_markers.id = 0;
        h_markers.type = visualization_msgs::Marker::SPHERE_LIST;
        h_markers.scale.x = h_markers.scale.y = h_markers.scale.z = 0.3;
        h_markers.color.r = 0.0; h_markers.color.g = 0.0; h_markers.color.b = 1.0; h_markers.color.a = 1.0;
        
        for (uint32_t id : h_node_ids_) {
            geometry_msgs::Point p;
            p.x = nodes_[id].pos.x(); p.y = nodes_[id].pos.y(); p.z = nodes_[id].pos.z();
            h_markers.points.push_back(p);
        }
        ma.markers.push_back(h_markers);

        // F-Nodes
        visualization_msgs::Marker f_markers;
        f_markers.header.frame_id = "world";
        f_markers.header.stamp = ros::Time::now();
        f_markers.ns = "f_nodes";
        f_markers.id = 1;
        f_markers.type = visualization_msgs::Marker::CUBE_LIST;
        f_markers.scale.x = f_markers.scale.y = f_markers.scale.z = 0.4;
        f_markers.color.r = 1.0; f_markers.color.g = 0.0; f_markers.color.b = 0.0; f_markers.color.a = 0.8;

        for (uint32_t id : f_node_ids_) {
            geometry_msgs::Point p;
            p.x = nodes_[id].pos.x(); p.y = nodes_[id].pos.y(); p.z = nodes_[id].pos.z();
            f_markers.points.push_back(p);
        }
        ma.markers.push_back(f_markers);

        // Edges
        visualization_msgs::Marker hh_edges, hf_edges;
        hh_edges.header.frame_id = hf_edges.header.frame_id = "world";
        hh_edges.header.stamp = hf_edges.header.stamp = ros::Time::now();
        hh_edges.ns = "hh_edges"; hf_edges.ns = "hf_edges";
        hh_edges.id = 2; hf_edges.id = 3;
        hh_edges.type = hf_edges.type = visualization_msgs::Marker::LINE_LIST;
        hh_edges.scale.x = hf_edges.scale.x = 0.05;
        
        // HH Edges: White
        hh_edges.color.r = 1.0; hh_edges.color.g = 1.0; hh_edges.color.b = 1.0; hh_edges.color.a = 0.5;
        // HF Edges: Yellow
        hf_edges.color.r = 1.0; hf_edges.color.g = 1.0; hf_edges.color.b = 0.0; hf_edges.color.a = 0.5;

        for (auto const& pair : nodes_) {
            for (auto const& edge : pair.second.edges) {
                if (nodes_.count(edge.neighbor_id)) {
                    geometry_msgs::Point p1, p2;
                    p1.x = pair.second.pos.x(); p1.y = pair.second.pos.y(); p1.z = pair.second.pos.z();
                    p2.x = nodes_[edge.neighbor_id].pos.x(); p2.y = nodes_[edge.neighbor_id].pos.y(); p2.z = nodes_[edge.neighbor_id].pos.z();
                    
                    if (edge.is_hf) {
                        hf_edges.points.push_back(p1);
                        hf_edges.points.push_back(p2);
                    } else {
                        // Only add HH edges once (from head to tail where head < tail) to avoid redundancy
                        if (pair.second.id < edge.neighbor_id) {
                            hh_edges.points.push_back(p1);
                            hh_edges.points.push_back(p2);
                        }
                    }
                }
            }
        }
        ma.markers.push_back(hh_edges);
        ma.markers.push_back(hf_edges);

        graph_pub_.publish(ma);
    }

    ros::NodeHandle nh_;
    ros::Subscriber nav_goal_sub_;
    ros::Publisher graph_pub_;
    ros::Publisher path_pub_;
    ros::WallTimer vis_timer_;

    std::string dtg_log_file_;
    std::unordered_map<uint32_t, ReaderNode> nodes_;
    std::vector<uint32_t> h_node_ids_;
    std::vector<uint32_t> f_node_ids_;
    std::vector<Eigen::Vector3d> points_;
};

} // namespace DTG

int main(int argc, char** argv) {
    ros::init(argc, argv, "dtg_reader_node");
    ros::NodeHandle nh("~");
    DTG::DTGReader reader(nh);
    ros::spin();
    return 0;
}
