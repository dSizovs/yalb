add_executable(milestone06 main.cpp)
target_link_libraries(milestone06 PRIVATE lib)

configure_file(plot_scaling.py plot_scaling.py COPYONLY)
configure_file(run_scaling.sh run_scaling.sh COPYONLY)
