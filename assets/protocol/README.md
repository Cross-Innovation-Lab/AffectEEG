# CEVB protocol metadata

These files contain labels, source-instance identities, source partitions,
condition partitions and integer evaluation-pair indices.

- `vocabulary.csv`: the original Mandarin 120-row `(group, one-based tone) ->
  pinyin` mapping. Code converts tone labels to zero-based integers.
- `seediv.json.gz` / `seedvii.json.gz`: portable exports of the fixed
  silent-articulation T2 protocol. Each contains `keys`, `key_splits`, `key_ids`,
  `content`, `emotion` and `evaluation_pairs`. Paths are relative to their raw
  corpus root. Gzip is ordinary lossless text compression; use Python's `gzip`
  module or `gzip -dc` to inspect the JSON.

The exports define the source-instance and target-key identifiers used by
the deterministic training sampler. Validation and test pairings specify
fixed source indices. A training epoch draws 100 pairs per training
key, traversing emotion donors without replacement before cycling, then
independently selecting content trials and interleaving keys.

| Track | Content trials | Emotion windows | Train / validation / test keys | Paired items per epoch / validation / test |
|---|---:|---:|---:|---:|
| SEED-IV | 4,800 | 37,575 | 127 / 38 / 37 | 12,700 / 3,800 / 3,700 |
| SEED-VII | 4,800 | 69,742 | 215 / 71 / 71 | 21,500 / 7,100 / 7,100 |

Content partitions contain 2,400 / 1,200 / 1,200 distinct source trials on each
track. Emotion partitions contain 26,295 / 5,655 / 5,625 windows on SEED-IV and
48,821 / 10,481 / 10,440 windows on SEED-VII. Source instances and condition
keys are disjoint across partitions; participants and original emotion trials
can occur in more than one partition.

Content `trial` is zero-based within the indicated tone-specific `.set` file.
SEED-IV `trial` is zero-based within the session and maps to MATLAB variable
`*_eeg{trial+1}`. SEED-VII `trial` is the official global trial number, 1–80,
matching the MATLAB key. `window_index` is zero-based and non-overlapping.
Both tracks use 800 raw samples at 200 Hz (four seconds).

The raw readers verify the recorded vocabulary, event sequence, channel order,
source labels and trial/window counts before a completed processed manifest is
written. Data releases that change these identities require corresponding protocol
identifiers and partitions.
