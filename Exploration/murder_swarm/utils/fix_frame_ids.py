#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import Image
# CONFIG
TARGET_FRAME = "world"
TOPIC_SUFFIX = "_world"

# List the topics and their message types here
# format: ("topic_name", MsgType)
# TOPICS = [
#     ("/odom_ugv_1", Odometry),
#     ("/odom_ugv_2", Odometry),
#     ("/point_cloud_ugv_1", PointCloud2),
#     ("/point_cloud_ugv_2", PointCloud2),
# ]

TOPICS = [
    ("/odom_uav", Odometry),
    ("/odom_ugv", Odometry),
    ("/point_cloud_uav", PointCloud2),
    ("/point_cloud_ugv", PointCloud2),
    ("/depth_uav", Image),
]

class FrameFixer(object):
    def __init__(self):
        self.pubs = {}
        for topic_name, msg_type in TOPICS:
            new_topic = topic_name + TOPIC_SUFFIX
            pub = rospy.Publisher(new_topic, msg_type, queue_size=10)
            self.pubs[topic_name] = pub
            # use lambda with default args to bind
            rospy.Subscriber(topic_name, msg_type, self.make_callback(topic_name, msg_type))
            rospy.loginfo("Subscribed to %s, will republish to %s", topic_name, new_topic)

    def make_callback(self, topic_name, msg_type):
        pub = self.pubs[topic_name]

        def callback(msg):
            # try to set header.frame_id if present
            if hasattr(msg, "header") and isinstance(msg.header, Header):
                msg.header.frame_id = TARGET_FRAME
            else:
                # some messages may have nested headers; add more cases here if needed
                pass

            pub.publish(msg)
        return callback


def main():
    rospy.init_node("fix_frame_ids", anonymous=True)
    FrameFixer()
    rospy.loginfo("frame_id fixer running. Target frame: %s, suffix: %s",
                  TARGET_FRAME, TOPIC_SUFFIX)
    rospy.spin()

if __name__ == "__main__":
    main()