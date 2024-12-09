#!/usr/bin/env python3

import rospy
import cv2
from ultralytics import YOLO
from math import floor
import numpy as np
import os
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError
import threading
from concurrent.futures import ThreadPoolExecutor

class ImageProcessorNode:
    def __init__(self):
        rospy.init_node('image_processor_node', anonymous=True)

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize YOLO models (Load them once at the start)
        rospy.loginfo("Loading YOLO models...")
        self.model = YOLO('yolov8n.pt')  # Object detection
        self.segmentation_model = YOLO("yolov8n-seg.pt")  # Segmentation
        rospy.loginfo("YOLO models loaded successfully")

        # Ensure the directories exist
        self.cropped_images_dir = './cropped_images'
        self.masks_dir = './masks'
        os.makedirs(self.cropped_images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

        # Define a variable to store the most recent image
        self.cv_image = None
        self.image_lock = threading.Lock()  # Thread lock for image access

        # Subscriber for the ROS image topic
        rospy.Subscriber('/pose/image', Image, self.image_callback)

        # Service to trigger image processing
        self.service = rospy.Service('/process_image', Trigger, self.handle_trigger)

        rospy.loginfo("Image Processor Node Initialized")

    def image_callback(self, msg):
        """Callback function to receive the ROS image."""
        try:
            # Convert ROS Image message to OpenCV image
            with self.image_lock:  # Use lock to ensure thread safety
                self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo("Image received and converted")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def handle_trigger(self, req):
        """Service handler to process the image when triggered."""
        with self.image_lock:
            if self.cv_image is None:
                rospy.logwarn("No image available to process")
                return TriggerResponse(success=False, message="No image available")

            # Clone the current image to avoid data race conditions
            current_image = self.cv_image.copy()

        rospy.loginfo("Processing image")

        # Process the received image
        bounding_boxes = self.process_image(current_image)
        self.crop_all_images(current_image, bounding_boxes)
        self.generate_masks(current_image, bounding_boxes)

        return TriggerResponse(success=True, message="Image processed")

    def process_image(self, cv_image):
        """Run object detection and return bounding boxes."""
        # Run object detection with YOLO model
        results = self.model(cv_image, classes=[0])  # Run detection without tracking

        bounding_boxes = []

        # Print bounding box coordinates for each detected person
        for result in results:
            for box in result.boxes:
                bb_coords_list = [int(floor(i)) for i in (list(box.xyxy)[0]).tolist()]
                x1, y1, x2, y2 = bb_coords_list

                bounding_boxes.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

                rospy.loginfo(f"Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        rospy.loginfo(f"Total Bounding Boxes: {len(bounding_boxes)}")
        return bounding_boxes

    def crop_image_with_points(self, points, image):
        """Crop the image based on given bounding box points."""
        # Ensure points are in the form [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Calculate the bounding box coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Crop the image using the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    def crop_all_images(self, image, bounding_boxes):
        """Crop all detected bounding boxes and save the images."""
        if not bounding_boxes:
            rospy.logwarn("No bounding boxes to crop")
            return

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.crop_single_image, image, box, i)
                for i, box in enumerate(bounding_boxes, start=1)
            ]

        for future in futures:
            future.result()

    def crop_single_image(self, image, box, index):
        """Crop a single image and save it."""
        cropped_image = self.crop_image_with_points(box, image)
        cropped_image_path = f'{self.cropped_images_dir}/cropped_image_{index}_RGB.png'
        cv2.imwrite(cropped_image_path, cropped_image)
        rospy.loginfo(f"Cropped image saved: {cropped_image_path}")

    def generate_masks(self, image, bounding_boxes):
        """Generate masks for each cropped image."""
        # Ensure the output directory exists
        os.makedirs('./crops_and_masks', exist_ok=True)

        if not bounding_boxes:
            rospy.logwarn("No bounding boxes for mask generation")
            return

        # Iterate over each bounding box
        for i, box in enumerate(bounding_boxes, start=1):
            # Crop the current image with bounding box
            cropped_image = self.crop_image_with_points(box, image)
            H, W, _ = cropped_image.shape

            # Predict masks using the YOLO segmentation model
            results = self.segmentation_model(cropped_image, classes=[0])

            # Iterate over each result
            for result in results:
                for j, mask in enumerate(result.masks.data):
                    # Move the mask to CPU and convert to numpy array
                    mask = mask.cpu().numpy() * 255

                    # Ensure the mask is of the correct type and reshape
                    mask = mask.astype(np.uint8)

                    # Resize the mask to the original image size
                    mask = cv2.resize(mask, (W, H))

                    # Construct the mask file name and save it
                    mask_filename = f'cropped_image_{i}_Mask.png'
                    mask_path = os.path.join(self.masks_dir, mask_filename)
                    cv2.imwrite(mask_path, mask)
                    rospy.loginfo(f"Mask saved successfully: {mask_path}")

                    # Move cropped image and mask to the ./crops_and_masks directory
                    cropped_image_path = f'{self.cropped_images_dir}/cropped_image_{i}_RGB.png'
                    os.rename(cropped_image_path, f'./crops_and_masks/cropped_image_{i}_RGB.png')
                    os.rename(mask_path, f'./crops_and_masks/{mask_filename}')

    def run(self):
        """Keep the node running."""
        rospy.spin()

if __name__ == '__main__':
    try:
        # Create an instance of the ImageProcessorNode class
        node = ImageProcessorNode()

        # Run the node
        node.run()
    except rospy.ROSInterruptException:
        pass
