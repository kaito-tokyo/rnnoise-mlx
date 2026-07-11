public struct FrameState: Sendable, Equatable {
    public static let featureCount = 65
    public static let convolutionHistoryFrames = 4

    public var featureHistory: [Float]
    public var gru1: [Float]
    public var gru2: [Float]
    public var gru3: [Float]

    public init(gruSize: Int) {
        precondition(gruSize > 0)
        featureHistory = .init(
            repeating: 0,
            count: Self.featureCount * Self.convolutionHistoryFrames
        )
        gru1 = .init(repeating: 0, count: gruSize)
        gru2 = .init(repeating: 0, count: gruSize)
        gru3 = .init(repeating: 0, count: gruSize)
    }

    public mutating func append(features: [Float]) {
        precondition(features.count == Self.featureCount)
        featureHistory.removeFirst(Self.featureCount)
        featureHistory.append(contentsOf: features)
    }
}

public struct FrameOutput: Sendable, Equatable {
    public var gains: [Float]
    public var vad: Float

    public init(gains: [Float], vad: Float) {
        precondition(gains.count == 32)
        self.gains = gains
        self.vad = vad
    }
}

/// Stable boundary between the RNNoise C DSP and the compiled BNNS graph.
public protocol FrameInferencing: AnyObject {
    var gruSize: Int { get }
    func process(features: [Float], state: inout FrameState) throws -> FrameOutput
}

