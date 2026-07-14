from rnnoise_mlx.tools.evaluate_corpus_capacity import evaluate


def test_evaluate_excludes_eval_speaker_and_checks_target():
    plan = {
        "official_sources": [],
        "regional_english_sources": [
            {"id": "eng", "eval_speakers": ["a:1"], "train_speaker_cap_seconds": 1800, "targets": {"E": 0.5, "B": 1.0, "C": 0.5}}
        ],
    }
    audit = {
        "source_summaries": {"eng": {}},
        "clips": [
            {"source_id": "eng", "speaker_id": "a:1", "edge_trimmed_duration_seconds": 900},
            {"source_id": "eng", "speaker_id": "a:2", "edge_trimmed_duration_seconds": 1800},
        ],
    }
    result = evaluate(plan, audit)["sources"]["eng"]
    assert result["edge_trimmed_train_seconds"] == 1800
    assert result["edge_trimmed_eval_seconds"] == 900
    assert result["designs"]["E"]["feasible_before_listening"]
    assert not result["designs"]["B"]["feasible_before_listening"]
