# cortex — an OS-inspired memory-hierarchy governor for LLM inference runtimes.
#
# The core is a standalone C++17 static library with no dependencies on the rest
# of the tree (it is pure policy logic), so it can be built, unit-tested and
# simulated independently of any backend.

add_library(cortex STATIC
    src/memory.cpp
    src/policy.cpp
)

target_include_directories(cortex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cortex PUBLIC cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cortex PRIVATE -Wall -Wextra)
endif()

if (LLAMA_BUILD_TESTS)
    enable_testing()

    add_executable(test-cortex tests/test-cortex.cpp)
    target_link_libraries(test-cortex PRIVATE cortex)
    target_compile_features(test-cortex PRIVATE cxx_std_17)

    add_test(NAME test-cortex COMMAND test-cortex)
endif()
