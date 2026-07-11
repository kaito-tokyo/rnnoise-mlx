import AVFoundation
import Foundation
import RNNoiseBNNS

enum CLIError: LocalizedError {
    case usage, conversion(String), model(Int32), processing(Int32)
    var errorDescription: String? {
        switch self {
        case .usage: return "usage: rnnoise-mlx-denoise [--probe input] | --model model.mlmodelc input output.wav"
        case .conversion(let value): return value
        case .model(let status): return "could not load BNNS model (status \(status))"
        case .processing(let status): return "BNNS/DSP processing failed (status \(status))"
        }
    }
}

let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000,
                                 channels: 1, interleaved: false)!

func probe(_ url: URL) throws {
    let file = try AVAudioFile(forReading: url), format = file.processingFormat
    print("path: \(url.path)")
    print("sample-rate: \(Int(format.sampleRate)) Hz")
    print("channels: \(format.channelCount)")
    print("frames: \(file.length)")
    print(String(format: "duration: %.3f s", Double(file.length) / format.sampleRate))
}

final class PCMWriter {
    private let file: AVAudioFile
    private var samples: [Float] = []
    init(url: URL) throws {
        do {
            file = try AVAudioFile(forWriting: url, settings: targetFormat.settings,
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        } catch { throw CLIError.conversion("open output: \(error)") }
    }
    func append(_ values: [Float], count: Int) throws {
        samples.append(contentsOf: values.prefix(count))
        if samples.count >= 48_000 { try flush() }
    }
    func flush() throws {
        guard !samples.isEmpty else { return }
        let buffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }
        do { try file.write(from: buffer) }
        catch { throw CLIError.conversion("write output: \(error)") }
        samples.removeAll(keepingCapacity: true)
    }
}

func denoise(modelURL: URL, inputURL: URL, outputURL: URL) throws {
    var processor: OpaquePointer?
    let loadStatus = rnnoise_bnns_processor_create(modelURL.path, &processor)
    guard loadStatus == RNNOISE_BNNS_OK, let processor else { throw CLIError.model(loadStatus) }
    defer { rnnoise_bnns_processor_destroy(processor) }

    let input: AVAudioFile
    do { input = try AVAudioFile(forReading: inputURL) }
    catch { throw CLIError.conversion("open input: \(error)") }
    guard let converter = AVAudioConverter(from: input.processingFormat, to: targetFormat) else {
        throw CLIError.conversion("AVFoundation cannot convert the input to 48 kHz mono float32")
    }
    let writer = try PCMWriter(url: outputURL)
    var pending: [Float] = []
    var processedFrames = 0
    var inputSamples = 0
    var outputSamples = 0
    var reachedEnd = false

    func processFrame(_ normalized: ArraySlice<Float>) throws {
        var frame = [Float](repeating: 0, count: 480)
        for (index, value) in normalized.enumerated() { frame[index] = value * 32_768 }
        var output = [Float](repeating: 0, count: 480), vad: Float = 0
        let status = rnnoise_bnns_process_audio_frame(processor, &frame, &output, &vad)
        guard status == RNNOISE_BNNS_OK else { throw CLIError.processing(status) }
        if processedFrames > 1 && outputSamples < inputSamples {
            for index in output.indices { output[index] /= 32_768 }
            let count = min(480, inputSamples - outputSamples)
            try writer.append(output, count: count)
            outputSamples += count
        }
        processedFrames += 1
    }

    while !reachedEnd {
        let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: 8_192)!
        var conversionError: NSError?
        let status = converter.convert(to: converted, error: &conversionError) { requested, inputStatus in
            if input.framePosition >= input.length {
                inputStatus.pointee = .endOfStream
                return nil
            }
            let buffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: requested)!
            do {
                try input.read(into: buffer, frameCount: requested)
                if buffer.frameLength == 0 { inputStatus.pointee = .endOfStream; return nil }
                inputStatus.pointee = .haveData; return buffer
            } catch { inputStatus.pointee = .endOfStream; return nil }
        }
        if let conversionError { throw CLIError.conversion("convert input: \(conversionError)") }
        if status == .error { throw CLIError.conversion("convert input: AVAudioConverter returned an error") }
        if converted.frameLength > 0, let channel = converted.floatChannelData?[0] {
            inputSamples += Int(converted.frameLength)
            pending.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
            while pending.count >= 480 {
                try processFrame(pending.prefix(480)); pending.removeFirst(480)
            }
        }
        reachedEnd = status == .endOfStream
    }

    if !pending.isEmpty { try processFrame(pending[...]) }
    let zeros = Array(repeating: Float(0), count: 480)
    while outputSamples < inputSamples { try processFrame(zeros[...]) }
    try writer.flush()
}

func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    if args == ["--help"] || args == ["-h"] {
        print("usage: rnnoise-mlx-denoise [--probe input] | --model model.mlmodelc input output.wav"); return
    }
    if args.count == 2, args[0] == "--probe" { try probe(URL(fileURLWithPath: args[1])); return }
    guard args.count == 4, args[0] == "--model" else { throw CLIError.usage }
    try denoise(modelURL: URL(fileURLWithPath: args[1]), inputURL: URL(fileURLWithPath: args[2]),
                outputURL: URL(fileURLWithPath: args[3]))
}

do { try run() }
catch { FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8)); exit(EXIT_FAILURE) }
