import sys, rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

topic, out = sys.argv[1], sys.argv[2]
rclpy.init()
n = Node("grabber")
br = CvBridge()
got = {}

def cb(msg):
    got['img'] = br.imgmsg_to_cv2(msg, "bgr8")

n.create_subscription(Image, topic, cb, 10)
for _ in range(60):
    if 'img' in got:
        break
    rclpy.spin_once(n, timeout_sec=0.5)

if 'img' in got:
    cv2.imwrite(out, got['img'])
    print("SAVED", out, got['img'].shape)
else:
    print("NO_IMAGE on", topic)
rclpy.shutdown()
