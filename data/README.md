# Data

This directory separates datasets and generated artifacts from the source code.

## `icews18/`

ICEWS18 is not distributed with this repository. The SERAG pipeline expects the
following benchmark files to be supplied locally:

- `entity2id.txt`: mapping from entity labels to integer identifiers;
- `relation2id.txt`: mapping from relation labels to integer identifiers;
- `train.txt`, `valid.txt`, and `test.txt`: temporal knowledge-graph splits.

Download these files from the
[ICEWS18 directory in the authors' RE-Net repository](https://github.com/INK-USC/RE-Net/tree/master/data/ICEWS18)
after reviewing the ICEWS terms below, then place them under `data/icews18/`.
The expected SHA-256 checksums for the benchmark variant used in the reported
experiments are:

| File | SHA-256 |
|---|---|
| `entity2id.txt` | `7d247def04f9e0a70b986da665fce498ef2057779842dd83af1df3df5c5e6d74` |
| `relation2id.txt` | `6099be14a4bb517019146818704858af040843df4caa77b816552223330dcb49` |
| `train.txt` | `019cfba096e15abae04e8d8d6f4dbeda2add15f53826681702f25298a9fbc7bf` |
| `valid.txt` | `aa9966e928768ad3a9fd28c7fe2c078458ec1caa523fc0ddf9d0a38ece349c50` |
| `test.txt` | `a00d46f993557cf78a8823b0c3a7b3b135f40ffa74150ee8f795794e2b536e33` |

The pipeline reads `train.txt` and creates train, validation, and test subsets
independently for each temporal snapshot. The original `valid.txt` and
`test.txt` files can be retained locally for future experiments using the
official benchmark splits.

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
