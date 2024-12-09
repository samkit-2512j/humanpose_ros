#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse

class ImageServiceNode:
    def __init__(self):
        rospy.init_node('image_service_node', anonymous=True)

        self.image = None

        # Subscriber for image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Service for image request
        self.image_service = rospy.Service('/request_image', Trigger, self.handle_image_request)

    def image_callback(self, msg):
        """Callback to store the latest image."""
        self.image = msg

    def handle_image_request(self, req):
        """Service handler to respond with the image."""
        if self.image is not None:
            # Publish the image only when requested
            rospy.loginfo("Image requested, sending image.")
            self.image_pub = rospy.Publisher('/pose/image', Image, queue_size=10)
            self.image_pub.publish(self.image)
            return TriggerResponse(success=True, message="Image published")
        else:
            return TriggerResponse(success=False, message="No image available")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ImageServiceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

