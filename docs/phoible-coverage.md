# PHOIBLE inventory coverage analysis

## Method and reproducibility

The primary source is the fixed PHOIBLE 2.0.1 CLDF release, DOI
[`10.5281/zenodo.2677911`](https://doi.org/10.5281/zenodo.2677911). The RNNoise
baseline comes from upstream
[`datasets.txt`](https://github.com/xiph/rnnoise/blob/main/datasets.txt), with
Hi-Fi TTS removed and LibriTTS-R English retained separately.

The comparison uses literal Unicode IPA strings. It does not fold by acoustic
similarity or distinctive features: `s`, `θ`, `ɕ`, `ʃ`, `sː`, and `sʰ` remain
different. Consonants, vowels, and tones are computed separately. Corpus size,
license, speaker count, and popularity do not enter the calculation.

Download PHOIBLE separately, then run:

```sh
python src/rnnoise_mlx/tools/analyze_phoible_coverage.py \
  /path/to/phoible/cldf \
  docs/phoible-baseline-selection.json \
  /tmp/phoible-analysis.json
```

The selection file records every baseline Inventory ID and every cumulative
scenario. The generated JSON contains all non-baseline PHOIBLE inventories;
it is intentionally not committed because it substantially reproduces the
licensed database. Repeating the command with the same CLDF release and
selection produces deterministic output and records a selection SHA-256.

## Baseline mapping result

The selection contains 26 corpus/language entries. With source uncertainty
preserved, the baseline covers:

| Class | Definite | Possible |
| --- | ---: | ---: |
| Consonants | 167 | 304 |
| Vowels | 55 | 199 |
| Tones | 2 | 10 |

“Definite” means that a segment occurs in every selected inventory for at least
one baseline entry. “Possible” means that it occurs in at least one selected
inventory. A candidate segment is therefore:

- **definitely novel** when absent from the possible baseline;
- **conditionally novel** when absent from the definite baseline but present in
  at least one alternative baseline inventory;
- **overlapping** when present in the definite baseline.

Important mapping limitations remain visible rather than being silently fixed:

- Sesotho is PHOIBLE inventory
  [`1017`](https://phoible.org/inventories/view/1017). In the 2.0.1 CLDF files,
  that contribution is named Sesotho and cites `sso_demuth1992`, but its
  language row is incorrectly linked to Sissano. The configuration records and
  excludes this known metadata mismatch explicitly.
- Setswana has no PHOIBLE 2.0.1 contribution. It is supplemented from Matlhaku
  and Rose’s source-specific consonant inventory and a seven-vowel description;
  their notation is retained literally. The source also warns that Setswana
  dialects differ in inventory.
- PHOIBLE has no exact match for each Latin American Spanish corpus, each
  British/Irish English accent corpus, or Nigerian English. Standard/regional
  proxies are marked as proxies in the selection file.
- Multiple inventories for Bengali, Basque, Catalan, Galician, and other
  languages remain separate source records.

Setswana support sources:

- K. Matlhaku and Y. Rose, “Phonetic Factors Affecting the Early Acquisition of
  Fricative and Affricate Consonants in Setswana,” including the 28-consonant
  table, labio-velarized series, allophony note, and dialect warning,
  [DOI `10.1177/01427237251404297`](https://doi.org/10.1177/01427237251404297).
- K. Matlhaku, “Nasal-Consonant Sequences in Setswana,” reporting seven phonemic
  vowels `/i ɪ ɛ a ɔ ʊ u/`,
  [open paper](https://www.mun.ca/linguistics/media/production/memorial/academic/faculty-of-humanities-and-social-sciences/linguistics/media-library/more/mlwpl/5_Matlhaku_MUNOPL_v3.pdf).

## Initial candidate differences

The tool scans all 2,953 non-baseline inventories. The table below is not a
ranking. It summarizes the union of definitely novel segments across every
PHOIBLE inventory for each previously discussed candidate; disagreement between
sources is preserved in the generated JSON.

| Language | Inventories | Definitely novel across source alternatives |
| --- | ---: | --- |
| Standard Arabic | 1 | 11 consonants, 2 vowels; includes `q`, pharyngeal/emphatic transcriptions and `ʕ̙` |
| Mandarin | 4 | 9 consonants, 14 vowels, 5 tones; no novel segment is common to all four inventories |
| German | 4 | 5 consonants, 2 vowels; includes `pf`, `ʀ/ʁ`, `ʏ/ʏː` depending on source |
| French | 4 | 5 consonants, 4 vowels; `ɥ` and `ø` are novel in all four inventories |
| Japanese | 3 | 4 consonants, 1 vowel; includes `ɴ`, `t̠ʃː`, `çː`, `ɯ̃`, depending on source |
| Georgian | 4 | 4 consonants, 1 vowel; includes `qʼ`, `qχʼ`, `t̪ʼ`, `ʁ` depending on source |
| Korean | 4 | 16 consonants, 14 vowels; source notation for the tense series differs substantially |
| Polish | 2 | 8 consonants, 3 vowels; `d̪z̪`, `t̪s̪`, and `ʑ` are novel in both inventories |
| Portuguese | 4 | 3 consonants, 35 vowels/diphthongs; Brazilian and European inventories differ strongly |
| Thai | 3 | 2 consonants, 4 vowels, 2 tones; tone encoding differs by inventory |
| Turkish | 4 | 7 consonants, 3 vowels; no definitely novel segment is common to all four inventories |
| Vietnamese | 4 | 7 vowels and 9 tones; no definitely novel consonant in these inventories |

The absence of a familiar language-level phoneme from this table does not mean
that the sound is absent from the language. For example, PHOIBLE analyses of
Japanese differ on whether sounds such as `[ɕ]` are represented as independent
phonemes, allophones, or parts of a more abstract analysis. The report therefore
uses Inventory IDs, not a synthesized “canonical” inventory. A detailed
phonological reference such as Laurence Labrune’s *The Phonology of Japanese*
is required before a disputed item affects language selection.

## Cumulative scenarios

These scenarios expose different complement patterns; they are not ranked and
do not imply corpus selection.

| Scenario | Literal additions beyond possible baseline |
| --- | --- |
| Japanese 197 → Beijing Mandarin 2457 → Arabic 2157 | Japanese: 2 C + 1 V; Mandarin: 5 C + 6 V; Arabic: 11 C + 2 V |
| Japanese 197 → Seoul Korean 2287 → Hanoi Vietnamese 2462 → Thai 27 | Japanese: 2 C + 1 V; Korean: 4 C + 7 V; Vietnamese: 2 V; Thai: 1 V + 1 tone |
| Standard Polish 2604 → Georgian 2428 → Standard Turkish 2416 | Polish: 7 C + 3 V; Georgian: 3 C; Turkish: 5 C + 1 V |
| French 2269 → Brazilian Portuguese 2207 → Standard German 2398 | French: 2 C + 1 V; Portuguese: 19 V/diphthongs; German: 4 C + 1 V |

PHOIBLE’s complete universe contains 2,029 consonant strings, 1,094 vowel
strings, and 60 tone strings. Treating this entire convenience sample as a
coverage target would reward extremely narrow transcription variants and is not
the goal. The generated JSON nevertheless records the remaining literal
segments after each scenario so that no custom similarity score is hidden.

## Interpretation boundary

This stage supports selection of a language set only. Before adoption, every
selection-driving segment should be checked against the PHOIBLE source or a
language-specific phonological description to determine whether it is a
contrastive phoneme, an allophone, dialect-specific, or loanword-specific.
Actual corpus phoneme frequency and transition coverage are a later analysis.
Audio corpus availability, quality, license, hours, and mixture weights remain
separate decisions.
