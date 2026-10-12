set(TARGET llama-moe-trace)
add_executable(${TARGET} moe-trace.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# end-to-end test; registered here because tests/ is configured before tools/
if (LLAMA_BUILD_TESTS AND TARGET test-llama-archs)
    find_program(LLAMA_MOE_TRACE_TEST_BASH NAMES bash)
    if (LLAMA_MOE_TRACE_TEST_BASH)
        set(MOE_TRACE_TEST_ARGS
            ${PROJECT_SOURCE_DIR}/tests/test-moe-trace.sh
            $<TARGET_FILE:test-llama-archs>
            $<TARGET_FILE:${TARGET}>)
        add_test(NAME test-moe-trace COMMAND ${LLAMA_MOE_TRACE_TEST_BASH} ${MOE_TRACE_TEST_ARGS})
        set_tests_properties(test-moe-trace PROPERTIES LABELS "main" TIMEOUT 300)
    endif()
endif()
