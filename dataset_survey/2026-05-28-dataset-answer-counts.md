# Dataset Answer/Class Counts vs Weak-Confidence Skew

Date: 2026-05-28

Question: is the weak-confidence skew explained by something as simple as the number of possible answers/classes?

## Short Takeaway

The number of possible answers/classes seems relevant, but it is not enough by itself.

- The most skewed datasets are all binary sentiment-style tasks: Amazon Polarity, SST-2, and twitter-sentiment.
- Dream and SciQ, which are originally multiple-choice QA datasets, have healthier weak-confidence distributions.
- PAWS is also binary, but it is much healthier than the sentiment datasets, so binary format alone cannot explain the skew.
- HellaSwag has four original choices, but the weak probe is near chance and low-confidence; this may be probe/task failure rather than a useful healthy distribution.

## Table

Sorted by `high_confidence_mass_gt_0_90`, the fraction of examples with weak confidence above 0.9.

| Dataset | Original Choices / Classes | Task Type | Probe Label Space | Weak Acc. | Median Conf. | Conf. > 0.9 | Skew Bucket |
|---|---:|---|---|---:|---:|---:|---|
| Amazon | 2 | sentiment classification | binary | 0.894 | 0.997 | 0.795 | very skewed/high-confidence |
| SST-2 | 2 | sentiment classification | binary | 0.842 | 0.977 | 0.681 | very skewed/high-confidence |
| Twitter Sentiment | 2 | tweet sentiment classification | binary | 0.768 | 0.908 | 0.526 | very skewed/high-confidence |
| BoolQ | 2 | binary yes/no QA | binary | 0.620 | 0.757 | 0.319 | moderately skewed or mixed |
| WiC | 2 | word-in-context meaning match | binary | 0.582 | 0.694 | 0.290 | moderately skewed or mixed |
| CoLA | 2 | linguistic acceptability | binary | 0.626 | 0.701 | 0.269 | moderately skewed or mixed |
| ANLI-R2 | 3 | NLI entailment/neutral/contradiction | binary entailment-vs-rest in our probe | 0.506 | 0.703 | 0.256 | moderately skewed or mixed |
| SciQ | 4 | multiple-choice science QA | binary candidate correctness | 0.621 | 0.660 | 0.224 | healthier / less skewed |
| PAWS | 2 | paraphrase/equivalence classification | binary | 0.634 | 0.607 | 0.177 | healthier / less skewed |
| Dream | 3 | multiple-choice dialogue QA | binary candidate correctness | 0.631 | 0.559 | 0.130 | healthier / less skewed |
| HellaSwag | 4 | multiple-choice continuation | binary candidate correctness | 0.504 | 0.358 | 0.031 | low-confidence / weak probe near chance |

## Interpretation

The current evidence supports a weak version of the answer-count hypothesis:

```text
fewer / simpler answer formats can make overconfident weak probes more likely,
especially for sentiment-style binary classification.
```

But the stronger claim is false:

```text
binary task does not automatically imply skewed confidence.
```

PAWS is binary but relatively healthy. BoolQ is binary but only moderately skewed. Dream and SciQ are originally multiple-choice and look healthier, but in our probe setup they are still converted into binary candidate-correctness labels. So the original answer count may matter through task structure and prompt richness, not simply through the final probe label dimension.

## Summary

The answer/class count seems related to the skew, but it does not fully explain it. The most skewed datasets are all binary sentiment-style tasks, while Dream and SciQ are healthier and originally multiple-choice. However, PAWS is binary and also healthy, so the task structure and shortcut availability probably matter at least as much as the number of labels.
