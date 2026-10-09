# CCA and HVSSE baselines

## Implementation status

These two baselines use the same `manifest.csv` files, splits, seeds, metrics,
and threshold-selection rule as SERAG and LoGNet. They are, however,
**adaptations to the ICEWS18 protocol**, not certified copies of the programs
that produced the published results.

| Method | Authors' code | Implementation in this repository |
|---|---|---|
| CCA | [Official repository](https://github.com/nju-websoft/CCA), GPL-3.0 | Independent reimplementation under the project's BSD license |
| HVSSE | No public repository identified as of October 6, 2026 | Independent reproduction based on the paper |

The official CCA code cannot be reused directly here. It targets FB15K-237 and
WN18RR, requires textual entity descriptions, pretrains a dataset-specific
BERT model, hard-codes several choices (including a 5% noise ratio), and lists
PyTorch 1.10.2 and Transformers 4.11.3 as dependencies. Copying GPL-3.0 code
into this BSD-licensed project would also introduce a license incompatibility.
The implementation in this repository was therefore written independently.

The HVSSE paper is published in a subscription journal, and no verifiable
public implementation was found. Details that are not specified in the public
paper are documented reproduction choices exposed as command-line options.

## Reproduced components

The CCA adaptation retains the following core ideas:

- head and tail reconstruction from textual and structural views;
- interactive contrast between the two spaces;
- fusion of reconstruction errors and cross-view disagreement;
- adaptive confidence recomputed without training labels.

ICEWS18 does not provide the entity descriptions required by the BERT model in
the original paper. By default, the textual view therefore encodes entity
labels with the same Sentence Transformer used by SERAG. An optional
`--entity-descriptions-file`, formatted as headerless TSV with
`entity_id<TAB>description`, can supply additional descriptions.

The HVSSE reproduction retains:

- positive and negative views based on entity co-occurrence in head and tail
  positions;
- a BiLSTM triple encoder;
- a sparse graph transformer enriched with head/tail degree and adjacency type;
- a multi-view contrastive loss and a TransE-style translational loss;
- a confidence score combining translational and cross-view consistency.

For both methods, structural context is derived only from training
observations. A validation or test observation may consult its training
neighbors, but it never becomes a neighbor of another observation. This rule
prevents future-information leakage in the chronological protocol.

## Running the baselines

After an editable installation (`python -m pip install -e .`), run:

```bash
./scripts/run_cca_hvsse_llm_cpu.sh
```

This launcher runs CCA and then HVSSE on `pooled_30_llm` and
`chronological_llm`, using training-anomaly ratios of 1%, 2.5%, 5%, 10%, and
20% and seeds 41–45. It reuses the exact manifests from the main experiments
and skips seeds that have already completed. The 20% ratio uses each
experiment's root manifest, consistently with the LoGNet and SERAG launchers.
The script produces one comparison file per ratio:

```text
outputs/cpu_llm_cca_hvsse/comparison_ratio_<ratio>.csv
```

Each table contains test AUROC, AUPRC, precision, recall, and F1 for LoGNet,
SERAG (`alpha = 0.5`), CCA, and HVSSE whenever their aggregate result files are
available.

Equivalent commands for individual runs are:

```bash
cca-train \
  --data-dir data/icews18 \
  --prepared-data-dir data/processed/icews18/experiments/pooled_30_llm/train_ratio_0.1 \
  --output-dir outputs/baselines/cca/pooled_30_llm_0.1 \
  --device cpu \
  --no-save-model \
  --seeds 41 42 43 44 45

hvsse-train \
  --data-dir data/icews18 \
  --prepared-data-dir data/processed/icews18/experiments/pooled_30_llm/train_ratio_0.1 \
  --output-dir outputs/baselines/hvsse/pooled_30_llm_0.1 \
  --device cpu \
  --no-save-model \
  --seeds 41 42 43 44 45
```

CCA computes a distribution over all 23,033 entities for each reconstruction,
so it is substantially more expensive than LoGNet on a CPU. A GPU is
recommended for complete experiments. Use `--score-batch-size` to control
memory consumption during evaluation.

## Interpreting the comparison

Scores produced in this repository are directly comparable **within the
adapted protocol**: the methods use the same examples, splits, positive class
for AUPRC, validation-based threshold selection, and no test-set optimization.

They are not direct reproductions of the papers' published tables:

- the papers study static graphs (notably FB15K-237 and WN18RR for CCA, and
  FB15K-237, WN18RR, NELL-995, FB15K, and UMLS for HVSSE), whereas this project
  aggregates or chronologically orders ICEWS18;
- CCA originally uses entity descriptions and a BERT model pretrained on the
  dataset; ICEWS18 does not provide this information;
- the papers train their detectors without supervision on a noisy knowledge
  graph and mainly report top-k metrics, whereas SERAG and LoGNet use labeled
  training anomalies;
- this project's LLM-generated anomalies do not exactly match CCA's random,
  semantically similar, and adversarial noise or HVSSE's corruption protocol.

The most transparent presentation therefore uses two separate tables:

1. results for all methods on the shared manifests, with each method's
   supervision budget stated explicitly;
2. a separate reproduction on the original datasets and noise types if the
   required data and entity descriptions are added.

## References

- Xiangyu Liu, Yang Liu, and Wei Hu, [*Knowledge Graph Error Detection with
  Contrastive Confidence Adaption*](https://arxiv.org/abs/2312.12108), AAAI 2024.
- Xinyue Xue et al., [*Heterogeneous Views and Spatial Structure Enhancement for
  triple error detection*](https://doi.org/10.1016/j.eswa.2024.124938), Expert
  Systems with Applications 256, 2024.
