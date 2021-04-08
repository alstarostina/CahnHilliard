cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project("projectswithdealii")

# find deal.ii package
find_package(deal.II 9.1.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../dealii/install $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

# add our executable
add_executable(main src/main.cpp)
deal_ii_setup_target(main)

add_executable( tutorial1 src/tutorial-1.cpp)
deal_ii_setup_target(tutorial1)

add_executable( tutorial3 src/tutorial-3.cpp)
deal_ii_setup_target(tutorial3)

add_executable( tutorial20 src/tutorial-20.cpp)
deal_ii_setup_target(tutorial20)

add_executable( CH src/CH.cpp)
deal_ii_setup_target(CH)


# create output folder
add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:main>/output)