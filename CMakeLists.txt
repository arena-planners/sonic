cmake_minimum_required(VERSION 3.12)
project(arena_planners_sonic NONE)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(PROJECTFILE pyproject.toml)

install(
  FILES
    planner.py
    planner.yaml
    srnn_net.py
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY gst_updated
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
