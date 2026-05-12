add_executable(milestone03 main.cpp)
target_link_libraries(milestone03 PRIVATE lib)

configure_file(plot.py plot.py COPYONLY)
