# ROS 2 Humble in Docker — runs on Raspberry Pi OS (Pi 5, arm64).
# The official ros images are multi-arch, so this pulls the arm64 variant on the Pi.
FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

# ROS + system deps for the sign detector, camera and lidar tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-v4l2-camera \
      ros-humble-vision-msgs \
      ros-humble-cv-bridge \
      ros-humble-image-transport \
      ros-humble-image-transport-plugins \
      python3-opencv \
      python3-numpy \
      python3-serial \
      python3-libgpiod \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# --- YDLIDAR X2: build the YDLidar-SDK, then fetch the ROS 2 driver ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/YDLIDAR/YDLidar-SDK.git /tmp/ydlidar-sdk \
    && cmake -S /tmp/ydlidar-sdk -B /tmp/ydlidar-sdk/build \
    && cmake --build /tmp/ydlidar-sdk/build -j2 \
    && cmake --install /tmp/ydlidar-sdk/build \
    && ldconfig \
    && rm -rf /tmp/ydlidar-sdk

# (optional) YOLO inference — heavy on arm; only needed for detector:=yolo
# RUN pip3 install --no-cache-dir ultralytics

WORKDIR /ros2_ws
# NOTE: master uses the pre-Humble declare_parameter() API and won't compile;
# the 'humble' branch has the corrected calls.
RUN git clone --depth 1 --branch humble https://github.com/YDLIDAR/ydlidar_ros2_driver.git \
      /ros2_ws/src/ydlidar_ros2_driver

# Bake the package into the image (self-contained / competition build).
# For live-editing, the compose file bind-mounts the source over this instead.
COPY . /ros2_ws/src/sign_detector

RUN source /opt/ros/humble/setup.bash \
    && rosdep update \
    && rosdep install --from-paths src -y --ignore-src \
    && MAKEFLAGS="-j2" colcon build --symlink-install --parallel-workers 2

RUN echo "source /opt/ros/humble/setup.bash"      >> /root/.bashrc \
    && echo "source /ros2_ws/install/setup.bash"   >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "sign_detector", "bringup.launch.py"]
