# RQ2 Interaction Recovery

Interaction coverage is the fraction of labelled passage pairs where both passages were retrieved in the top-k context. Sign recovery is computed only over those evaluable pairs.

| Setting | Target | Type | Total labelled pairs | Evaluable pairs | Coverage | Correct signs | Sign recovery |
|---|---|---:|---:|---:|---:|---:|---:|
| BGE-base + lexical | Gold/reference answer | complementary | 20 | 17 | 0.850 | 12 | 0.706 |
| BGE-base + lexical | Gold/reference answer | redundant | 22 | 18 | 0.818 | 18 | 1.000 |
| BGE-base + lexical | Gold/reference answer | conflicting | 18 | 18 | 1.000 | 18 | 1.000 |
| BGE-base + lexical | Gold/reference answer | overall | 60 | 53 | 0.883 | 48 | 0.906 |
| BGE-base + target LL | Gold/reference answer | complementary | 20 | 17 | 0.850 | 9 | 0.529 |
| BGE-base + target LL | Gold/reference answer | redundant | 22 | 18 | 0.818 | 17 | 0.944 |
| BGE-base + target LL | Gold/reference answer | conflicting | 18 | 18 | 1.000 | 13 | 0.722 |
| BGE-base + target LL | Gold/reference answer | overall | 60 | 53 | 0.883 | 39 | 0.736 |
| BGE-base + target LL | Generated answer | complementary | 20 | 17 | 0.850 | 10 | 0.588 |
| BGE-base + target LL | Generated answer | redundant | 22 | 18 | 0.818 | 8 | 0.444 |
| BGE-base + target LL | Generated answer | conflicting | 18 | 18 | 1.000 | 7 | 0.389 |
| BGE-base + target LL | Generated answer | overall | 60 | 53 | 0.883 | 25 | 0.472 |
| BGE-base + contrastive LL | Gold/reference answer | complementary | 20 | 17 | 0.850 | 8 | 0.471 |
| BGE-base + contrastive LL | Gold/reference answer | redundant | 22 | 18 | 0.818 | 17 | 0.944 |
| BGE-base + contrastive LL | Gold/reference answer | conflicting | 18 | 18 | 1.000 | 12 | 0.667 |
| BGE-base + contrastive LL | Gold/reference answer | overall | 60 | 53 | 0.883 | 37 | 0.698 |
| BGE-base + contrastive LL | Generated answer | complementary | 20 | 17 | 0.850 | 9 | 0.529 |
| BGE-base + contrastive LL | Generated answer | redundant | 22 | 18 | 0.818 | 9 | 0.500 |
| BGE-base + contrastive LL | Generated answer | conflicting | 18 | 18 | 1.000 | 6 | 0.333 |
| BGE-base + contrastive LL | Generated answer | overall | 60 | 53 | 0.883 | 24 | 0.453 |
| TF-IDF + contrastive LL | Gold/reference answer | complementary | 20 | 14 | 0.700 | 6 | 0.429 |
| TF-IDF + contrastive LL | Gold/reference answer | redundant | 22 | 12 | 0.545 | 11 | 0.917 |
| TF-IDF + contrastive LL | Gold/reference answer | conflicting | 18 | 15 | 0.833 | 10 | 0.667 |
| TF-IDF + contrastive LL | Gold/reference answer | overall | 60 | 41 | 0.683 | 27 | 0.659 |
| TF-IDF + contrastive LL | Generated answer | complementary | 20 | 14 | 0.700 | 7 | 0.500 |
| TF-IDF + contrastive LL | Generated answer | redundant | 22 | 12 | 0.545 | 7 | 0.583 |
| TF-IDF + contrastive LL | Generated answer | conflicting | 18 | 15 | 0.833 | 6 | 0.400 |
| TF-IDF + contrastive LL | Generated answer | overall | 60 | 41 | 0.683 | 20 | 0.488 |
