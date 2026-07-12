// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RNNoiseMLX",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "RNNoiseBNNS", targets: ["RNNoiseBNNS"]),
        .executable(name: "rnnoise-mlx-denoise", targets: ["RNNoiseMLXDenoise"]),
        .executable(name: "rnnoise-bnns-runner", targets: ["RNNoiseBNNSRunner"]),
    ],
    targets: [
        .target(
            name: "RNNoiseBNNS",
            dependencies: ["RNNoiseDSP"],
            path: "Sources/RNNoiseBNNS",
            publicHeadersPath: "include",
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "RNNoiseDSP",
            path: "training/vendor/xiph-rnnoise",
            sources: [
                "src/denoise.c", "src/pitch.c", "src/celt_lpc.c", "src/kiss_fft.c",
                "src/parse_lpcnet_weights.c", "src/rnnoise_tables.c", "src/training_globals.c",
            ],
            publicHeadersPath: "include",
            cSettings: [
                .define("TRAINING", to: "1"),
                .headerSearchPath("."),
                .headerSearchPath("src"),
            ]
        ),
        .testTarget(
            name: "RNNoiseBNNSTests",
            dependencies: ["RNNoiseBNNS"],
            path: "Tests/RNNoiseBNNSTests"
        ),
        .executableTarget(
            name: "RNNoiseMLXDenoise",
            dependencies: ["RNNoiseBNNS"],
            path: "Sources/RNNoiseMLXDenoise",
            linkerSettings: [.linkedFramework("AVFoundation")]
        ),
        .executableTarget(
            name: "RNNoiseBNNSRunner",
            dependencies: ["RNNoiseBNNS"],
            path: "training/tests",
            sources: ["bnns_runner.c"]
        ),
    ]
)
