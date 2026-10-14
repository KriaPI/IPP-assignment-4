add_executable(hello 
    hello.cpp
)

add_executable(ex1send 
    exercise1send.cpp
)

add_executable(ex1bcast 
    exercise1bcast.cpp
)

target_link_libraries(hello MPI::MPI_CXX)  
target_link_libraries(ex1send MPI::MPI_CXX)   
target_link_libraries(ex1bcast MPI::MPI_CXX)   