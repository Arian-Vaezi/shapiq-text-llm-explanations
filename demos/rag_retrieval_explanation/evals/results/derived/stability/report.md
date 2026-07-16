# Controlled Attribution Stability

Stability is computed from completed controlled-run artifacts without
re-running retrieval, generation, or likelihood scoring. Correlations rank
chunks by absolute first-order Shapley attribution.

## bge_contrastive_ll vs tfidf_contrastive_ll (generated)

- Cases: 50
- Mean retrieved-set Jaccard: 0.600
- Top-attribution agreement: 0.780
- Mean Spearman on common retrieved chunks: 0.849
- Mean Spearman on union with missing chunks as zero: 0.389

## bge_contrastive_ll vs tfidf_contrastive_ll (gold)

- Cases: 50
- Mean retrieved-set Jaccard: 0.600
- Top-attribution agreement: 0.760
- Mean Spearman on common retrieved chunks: 0.819
- Mean Spearman on union with missing chunks as zero: 0.340

## bge_lexical vs bge_contrastive_ll (gold)

- Cases: 50
- Mean retrieved-set Jaccard: 1.000
- Top-attribution agreement: 0.580
- Mean Spearman on common retrieved chunks: 0.458
- Mean Spearman on union with missing chunks as zero: 0.458

## bge_lexical vs bge_target_ll (gold)

- Cases: 50
- Mean retrieved-set Jaccard: 1.000
- Top-attribution agreement: 0.580
- Mean Spearman on common retrieved chunks: 0.463
- Mean Spearman on union with missing chunks as zero: 0.463

## bge_target_ll vs bge_contrastive_ll (generated)

- Cases: 50
- Mean retrieved-set Jaccard: 1.000
- Top-attribution agreement: 1.000
- Mean Spearman on common retrieved chunks: 0.993
- Mean Spearman on union with missing chunks as zero: 0.993

## bge_target_ll vs bge_contrastive_ll (gold)

- Cases: 50
- Mean retrieved-set Jaccard: 1.000
- Top-attribution agreement: 0.980
- Mean Spearman on common retrieved chunks: 0.936
- Mean Spearman on union with missing chunks as zero: 0.936
