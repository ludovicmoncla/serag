# Baselines CCA et HVSSE

## Statut des implémentations

Ces deux baselines utilisent les mêmes `manifest.csv`, partitions, graines,
métriques et règle de sélection du seuil que SERAG et LoGNet. Elles sont
cependant des **adaptations au protocole ICEWS18**, et non des copies certifiées
des programmes ayant produit les résultats publiés.

| Méthode | Code des auteurs | Implémentation dans ce dépôt |
|---|---|---|
| CCA | [Dépôt officiel](https://github.com/nju-websoft/CCA), GPL-3.0 | Réimplémentation indépendante sous la licence BSD du projet |
| HVSSE | Aucun dépôt public identifié au 6 octobre 2026 | Reproduction indépendante à partir de l'article |

Le code officiel CCA n'est pas directement réutilisable ici. Il cible
FB15K-237 et WN18RR, demande les descriptions textuelles des entités, préentraîne
un BERT propre au jeu de données, fixe plusieurs choix dans le code (dont le
ratio de bruit à 5 %) et annonce PyTorch 1.10.2 et Transformers 4.11.3. Copier ce
code GPL-3.0 dans le projet BSD poserait en outre une contrainte de licence. La
présente version a donc été écrite indépendamment.

L'article HVSSE est publié dans une revue à accès fermé et aucune implémentation
publique vérifiable n'a été trouvée. Les détails non spécifiés publiquement sont
des choix de reproduction documentés par les options de ligne de commande.

## Ce qui est reproduit

CCA conserve les idées centrales suivantes :

- reconstruction de la tête et de la queue depuis une vue textuelle et une vue
  structurelle ;
- contraste interactif entre les deux espaces ;
- fusion des erreurs de reconstruction et du désaccord entre vues ;
- confiance adaptative, recalculée sans utiliser les étiquettes d'entraînement.

ICEWS18 ne fournit pas les descriptions d'entités nécessaires au BERT de
l'article. Par défaut, la vue textuelle encode donc les libellés avec le même
Sentence-Transformer que SERAG. Un fichier optionnel
`--entity-descriptions-file`, au format TSV sans en-tête
`entity_id<TAB>description`, permet d'ajouter des descriptions.

HVSSE conserve :

- les vues positives et négatives fondées sur la cooccurrence des entités en
  position tête et queue ;
- un encodeur BiLSTM des triplets ;
- un Transformer de graphe parcimonieux enrichi par le degré tête/queue et le
  type d'adjacence ;
- une perte contrastive multi-vues et une perte translationnelle de type
  TransE ;
- une confiance combinant cohérence translationnelle et cohérence des vues.

Pour les deux méthodes, seules les observations d'entraînement peuvent fournir
du contexte structurel. Une observation de validation ou de test peut consulter
ses voisines d'entraînement, mais ne devient jamais la voisine d'une autre
observation. Cette convention évite une fuite du futur dans le protocole
chronologique.

## Exécution

Après une installation éditable (`python -m pip install -e .`) :

```bash
./scripts/run_cca_hvsse_llm_cpu.sh
```

Ce lanceur exécute CCA puis HVSSE sur `pooled_30_llm` et
`chronological_llm`, avec 1 %, 2,5 %, 5 %, 10 % et 20 % d'anomalies
d'entraînement et les graines 41 à 45. Il réutilise exactement les manifests des
expériences principales et reprend les graines déjà terminées. Le ratio 20 %
utilise le manifeste racine de chaque expérience, conformément aux lanceurs
LoGNet et SERAG. Le script produit un fichier par ratio :

```text
outputs/cpu_llm_cca_hvsse/comparison_ratio_<ratio>.csv
```

Chaque tableau rassemble AUROC, AUPRC, précision, rappel et F1 de test pour
LoGNet, SERAG (`alpha = 0.5`), CCA et HVSSE lorsque leurs fichiers agrégés
sont disponibles.

Les commandes individuelles équivalentes sont :

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

CCA calcule une distribution sur les 23 033 entités pour chaque reconstruction.
Il est donc nettement plus coûteux que LoGNet sur CPU. Une GPU est recommandée
pour les expériences complètes. `--score-batch-size` permet de contrôler la
mémoire utilisée lors de l'évaluation.

## Interprétation de la comparaison

Les scores produits dans ce dépôt sont directement comparables **dans le
protocole adapté** : mêmes exemples, mêmes partitions, même classe positive pour
l'AUPRC, même sélection du seuil sur validation et aucune optimisation sur le
test.

Ils ne constituent pas une reproduction directe des tableaux des articles :

- les articles étudient des graphes statiques (notamment FB15K-237 et WN18RR
  pour CCA ; FB15K-237, WN18RR, NELL-995, FB15K et UMLS pour HVSSE), alors que
  ce projet agrège ou ordonne temporellement ICEWS18 ;
- CCA emploie à l'origine des descriptions d'entités et un BERT préentraîné sur
  le jeu de données ; cette information n'existe pas dans ICEWS18 ;
- les articles entraînent leurs détecteurs de manière non supervisée sur un KG
  bruité et rapportent principalement des mesures top-k, tandis que SERAG et
  LoGNet utilisent les anomalies étiquetées du train ;
- les anomalies LLM du projet ne suivent pas exactement les bruits aléatoire,
  sémantiquement similaire et adversarial de CCA, ni le protocole de corruption
  de HVSSE.

La comparaison la plus honnête consiste donc à présenter deux tableaux :

1. les résultats de toutes les méthodes sur les manifests communs, en indiquant
   le budget de supervision propre à chacune ;
2. séparément, une reproduction sur les jeux et types de bruit originaux si les
   données et descriptions nécessaires sont ajoutées.

## Références

- Xiangyu Liu, Yang Liu et Wei Hu, [*Knowledge Graph Error Detection with
  Contrastive Confidence Adaption*](https://arxiv.org/abs/2312.12108), AAAI 2024.
- Xinyue Xue et al., [*Heterogeneous Views and Spatial Structure Enhancement for
  triple error detection*](https://doi.org/10.1016/j.eswa.2024.124938), Expert
  Systems with Applications 256, 2024.
