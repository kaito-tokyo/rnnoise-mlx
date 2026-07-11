// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RNNoiseBNNS",
    platforms: [.macOS(.v15)],
    products: [.library(name: "RNNoiseBNNS", targets: ["RNNoiseBNNS"])],
    targets: [
        .target(name: "RNNoiseBNNS"),
        .testTarget(name: "RNNoiseBNNSTests", dependencies: ["RNNoiseBNNS"]),
    ]
)

