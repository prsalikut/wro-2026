#!/usr/bin/env bash
set -e
docker run -it --rm \
  --name sign_detector \
  --network host \
  --ipc host \
  --privileged \
  -v /dev:/dev \
  sign_detector:humble \
  ros2 launch sign_detector bringup.launch.py
