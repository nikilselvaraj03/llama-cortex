# cortex governor: reads the real device/model and emits the memory plan

set(TARGET llama-cortex)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common cortex ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
