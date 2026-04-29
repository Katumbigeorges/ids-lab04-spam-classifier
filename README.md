# IDS Lab 04 — Anomaly Detection and Machine Learning
**04-625 A: Intrusion Detection Systems | Carnegie Mellon University Africa**

## Overview
This project implements two machine-learning classifiers to detect spam emails:
1. **Bayesian Classifier** — Naive Bayes with MAP and Maximum Likelihood methods
2. **Single-Neuron Neural Network** — trained with batch gradient descent

---

## File Structure
```
lab04/
├── main.py                    # Run all three parts
├── part1_data_generation.py   # Part I  – synthetic dataset generation
├── part2_bayesian.py          # Part II – Bayesian (MAP + ML) classifier
├── part3_neural_network.py    # Part III – single-neuron NN classifier
├── spam_dict.txt              # Spam word dictionary (90 words)
├── ham_dict.txt               # Ham  word dictionary (90 words)
├── spam_emails.txt            # Generated (after running part1)
├── ham_emails.txt             # Generated (after running part1)
└── README.md
```

---

## Requirements
- Python 3.10+
- **No external libraries required** — uses only the Python standard library (`math`, `random`, `collections`)

---

## How to Run

### Option A — Run everything at once
```bash
python main.py
```

### Option B — Run parts individually
```bash
# Step 1: Generate the dataset
python part1_data_generation.py

# Step 2: Train and test the Bayesian classifier
python part2_bayesian.py

# Step 3: Train and test the Neural Network classifier
python part3_neural_network.py
```

### Option C — Google Colab
1. Upload all `.py` and `.txt` files to your Colab session (or clone the repo)
2. Run cells in order: Part 1 → Part 2 → Part 3

---

## Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `ps`      | 0.75  | P(spam word \| spam email) |
| `qs`      | 0.20  | P(spam word \| ham email)  |
| `l_min`   | 5     | Minimum email length (words) |
| `l_max`   | 15    | Maximum email length (words) |
| N spam    | 1 000 | Training spam emails |
| N ham     | 2 300 | Training ham emails  |

---

## Results Summary
| Classifier | Method | Accuracy on 10 test emails |
|------------|--------|---------------------------|
| Bayesian   | MAP    | See output                |
| Bayesian   | ML     | See output                |
| Neural Net | Threshold on x_ham | See output     |

---

## Key Design Decisions
- **Laplace smoothing** prevents zero-probability issues for unseen words in the Bayesian model.
- **Log-space arithmetic** avoids floating-point underflow when multiplying many small probabilities.
- **Batch gradient descent** updates weights once per epoch using the full training set.
- The NN threshold `x_T` is derived analytically from the condition `y(1) − y(x_T) = y(x_T) − y(0)`.
