cmake_minimum_required(VERSION 3.25)
project(rnnoise_mlx LANGUAGES C)

add_library(rnnoise_dsp STATIC
    Vendors/xiph-rnnoise/src/denoise.c
    Vendors/xiph-rnnoise/src/pitch.c
    Vendors/xiph-rnnoise/src/celt_lpc.c
    Vendors/xiph-rnnoise/src/kiss_fft.c
    Vendors/xiph-rnnoise/src/parse_lpcnet_weights.c
    Vendors/xiph-rnnoise/src/rnnoise_tables.c
    Vendors/xiph-rnnoise/src/training_globals.c
)
target_compile_definitions(rnnoise_dsp PRIVATE TRAINING=1)
target_compile_features(rnnoise_dsp PUBLIC c_std_11)
target_include_directories(rnnoise_dsp PUBLIC Vendors/xiph-rnnoise/include PRIVATE Vendors/xiph-rnnoise Vendors/xiph-rnnoise/src)

add_library(rnnoise_bnns STATIC
    Sources/RNNoiseBNNS/rnnoise_bnns.c
)
add_library(RNNoiseMLX::BNNS ALIAS rnnoise_bnns)

target_compile_features(rnnoise_bnns PUBLIC c_std_11)
target_link_libraries(rnnoise_bnns PUBLIC rnnoise_dsp)
target_include_directories(rnnoise_bnns
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Sources/RNNoiseBNNS/include>
        $<INSTALL_INTERFACE:include>
)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(rnnoise_bnns PUBLIC "${ACCELERATE_FRAMEWORK}")
else()
    message(FATAL_ERROR "RNNoise BNNS inference requires Apple Accelerate")
endif()

include(GNUInstallDirs)
install(TARGETS rnnoise_bnns rnnoise_dsp
    EXPORT RNNoiseBNNSTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES Sources/RNNoiseBNNS/include/rnnoise_bnns.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
