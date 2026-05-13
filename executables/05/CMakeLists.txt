add_executable(milestone05 main.cpp)
target_link_libraries(milestone05 PRIVATE lib)

configure_file(plot.py plot.py COPYONLY)
configure_file(ghia_data.py ghia_data.py COPYONLY)
