#!/usr/bin/bash

while true
do
	rosservice call /request_image
	# sleep .05
	# rosservice call /request_image
	# sleep .05
	rosservice call /process_image
	# sleep .05
	rosservice call /pose/estimate "{}"
	sleep .05
	rm ~/pose/pose_ros/cropped_images/* ~/pose/pose_ros/crops_and_masks/* ~/pose/pose_ros/masks/*
done