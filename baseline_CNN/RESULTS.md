# Baseline CNN — Test Results

Checkpoint: `checkpoints/best.pt` (epoch 37/40, val_acc 0.6655)
Evaluated with `python test.py` on `data/test` (7178 images, untouched during training).

## What the metrics mean

- **Accuracy** — fraction of all test images classified correctly. Misleading alone here because classes are imbalanced (e.g. `disgust` is only ~1.5% of the data, `happy` is ~25%) — a model that's bad at rare classes can still post high accuracy.
- **F1 score** (per class) — harmonic mean of precision and recall for that one emotion. Precision = "of the images I called angry, how many really were angry". Recall = "of the truly angry images, how many did I catch". F1 punishes a model that's only good at one of the two.
- **Macro F1** — F1 averaged across the 7 classes, all weighted equally. The fair measure when you care about rare classes (disgust) as much as common ones (happy).
- **Weighted F1** — F1 averaged across classes, weighted by how many test examples each class has. Closer to overall accuracy since it's dominated by the big classes.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.6648 |
| Macro F1 | 0.6469 |
| Weighted F1 | 0.6625 |

### Per-class

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.5792 | 0.6106 | 0.5945 | 958 |
| disgust | 0.6271 | 0.6667 | 0.6463 | 111 |
| fear | 0.5464 | 0.4082 | 0.4673 | 1024 |
| happy | 0.8747 | 0.8540 | 0.8642 | 1774 |
| neutral | 0.6128 | 0.6456 | 0.6288 | 1233 |
| sad | 0.5186 | 0.5597 | 0.5384 | 1247 |
| surprise | 0.7555 | 0.8255 | 0.7890 | 831 |

`fear` is the weakest class (F1 0.47) — mostly confused with `sad` and `angry` (see confusion matrix below). `happy` and `surprise` are the strongest, likely the most visually distinct expressions.

### Confusion matrix (rows = true, cols = predicted)

|          | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| **angry** | 585 | 17 | 83 | 32 | 91 | 131 | 19 |
| **disgust** | 25 | 74 | 4 | 2 | 1 | 4 | 1 |
| **fear** | 131 | 6 | 418 | 30 | 95 | 239 | 105 |
| **happy** | 33 | 4 | 30 | 1515 | 82 | 54 | 56 |
| **neutral** | 69 | 6 | 62 | 77 | 796 | 203 | 20 |
| **sad** | 146 | 8 | 109 | 47 | 218 | 698 | 21 |
| **surprise** | 21 | 3 | 59 | 29 | 16 | 17 | 686 |
