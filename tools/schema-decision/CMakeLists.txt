# the engine: used by llama-schema-decision and by llama-server's /v1/decision endpoint
add_library(llama-schema-decision-engine STATIC schema-decision.cpp schema-decision.h)
target_include_directories(llama-schema-decision-engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-schema-decision-engine PUBLIC llama-common llama)
target_compile_features(llama-schema-decision-engine PUBLIC cxx_std_17)
# llama-server links it into a shared library
set_target_properties(llama-schema-decision-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGET llama-schema-decision)
add_executable(${TARGET} schema-decision-cli.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama-schema-decision-engine llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# tests; registered here because tests/ is configured before tools/
if (LLAMA_BUILD_TESTS)
    add_executable(test-schema-decision ${PROJECT_SOURCE_DIR}/tests/test-schema-decision.cpp)
    target_link_libraries(test-schema-decision PRIVATE llama-schema-decision-engine llama-common llama)
    target_compile_features(test-schema-decision PRIVATE cxx_std_17)
    # schema and tokenization checks on the vocabularies in models/
    add_test(NAME test-schema-decision-vocab
             COMMAND test-schema-decision --vocab ${PROJECT_SOURCE_DIR}/models)
    set_tests_properties(test-schema-decision-vocab PROPERTIES LABELS "main" TIMEOUT 300)
    if (TARGET test-llama-archs)
        find_program(LLAMA_SCHEMA_DECISION_TEST_BASH NAMES bash)
        if (LLAMA_SCHEMA_DECISION_TEST_BASH)
            add_test(NAME test-schema-decision
                     COMMAND ${LLAMA_SCHEMA_DECISION_TEST_BASH} ${PROJECT_SOURCE_DIR}/tests/test-schema-decision.sh
                             $<TARGET_FILE:test-llama-archs> $<TARGET_FILE:test-schema-decision>)
            set_tests_properties(test-schema-decision PROPERTIES LABELS "main" TIMEOUT 900)
        endif()
    endif()
endif()
