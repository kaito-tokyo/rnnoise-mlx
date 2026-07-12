import Testing
import RNNoiseBNNS

@Test func cStateAllocatesAndAppends() {
    var state = RNNoiseBNNSFrameState()
    rnnoise_bnns_state_init(&state)
    rnnoise_bnns_state_reset(&state)
}
