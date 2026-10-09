# Data

This directory separates datasets and generated artifacts from the source code.

## `icews18/`

The SERAG pipeline expects the following ICEWS18 benchmark files:

- `entity2id.txt`: mapping from entity labels to integer identifiers;
- `relation2id.txt`: mapping from relation labels to integer identifiers;
- `train.txt`, `valid.txt`, and `test.txt`: temporal knowledge-graph splits.

The pipeline reads `train.txt` and creates train, validation, and test subsets
independently for each temporal snapshot. The original `valid.txt` and
`test.txt` files are retained for future experiments using the official
benchmark splits.

Run `serag-describe-data` from the repository root to validate all three files
and generate readable tables, descriptive statistics, and the degree-
distribution figure under `data/processed/icews18/`. These derived files are
reproducible and therefore excluded from version control.

When LLM anomalies are generated, reusable files are stored under
`data/processed/icews18/llm_anomalies/`. Each `snapshot_NNN.csv` contains the
text labels and numeric IDs required to evaluate LoGNet and SERAG on exactly
the same negative triples. These generated files are ignored by Git and should
be archived separately when they form part of an experiment.

Prepared paper experiments are written under
`data/processed/icews18/experiments/<protocol>/` by `serag-prepare`. Each
directory contains a checksummed `manifest.csv` shared by SERAG and LoGNet and
a `metadata.json` file recording the protocol, seed, timestamps, and class
counts.

## Dataset provenance and terms

ICEWS18 is derived from the [ICEWS Coded Event Data](https://doi.org/10.7910/DVN/28075)
and follows the daily temporal-knowledge-graph benchmark introduced for
recurrent event-network research. Cite both the ICEWS dataset and the benchmark
paper when using these files:

- Elizabeth Boschee et al., *ICEWS Coded Event Data*, Harvard Dataverse,
  <https://doi.org/10.7910/DVN/28075>.
- Woojeong Jin et al., [*Recurrent Event Network for Reasoning over Temporal
  Knowledge Graphs*](https://arxiv.org/abs/1904.05530), 2020.

The dataset is **not covered by this repository's BSD-2-Clause license**. The
official ICEWS terms limit use and copying to research and educational purposes,
prohibit commercial use, and state that users may wish to contact the dataset
maintainer before use. Consult the current terms on Harvard Dataverse before
redistributing or using the data. Source code in this repository remains under
the BSD-2-Clause license.

## `snapshots/`

If serialized graph snapshots are added, document their schema, provenance,
and relationship to ICEWS18 before using them in the public pipeline. Generated
snapshots and experiment outputs should remain outside version control unless
they are intentionally released with complete provenance and licensing notes.
