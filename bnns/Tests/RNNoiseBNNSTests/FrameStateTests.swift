import Testing
@testable import RNNoiseBNNS

@Test func historyKeepsFourFrames() {
    var state = FrameState(gruSize: 8)
    for value in 1...5 {
        state.append(features: .init(repeating: Float(value), count: 65))
    }
    #expect(state.featureHistory.count == 260)
    #expect(state.featureHistory.first == 2)
    #expect(state.featureHistory.last == 5)
}

