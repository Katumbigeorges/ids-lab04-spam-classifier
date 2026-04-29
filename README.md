# IDS Lab 04 — Anomaly Detection and Machine Learning
**04-625 A: Intrusion Detection Systems | Carnegie Mellon University Africa**

## Overview
This project implements two machine-learning classifiers to detect spam emails:
1. **Bayesian Classifier** — Naive Bayes with MAP and Maximum Likelihood methods
2. **Single-Neuron Neural Network** — trained with batch gradient descent

---

## File Structure
ids-lab04-spam-classifier/
├── part1_dataset.ipynb     # All three parts: data generation, Bayesian, Neural Network
├── spam_dict.txt           # Spam word dictionary (53 words)
├── ham_dict.txt            # Ham word dictionary (53 words)
├── spam_emails.txt         # 1000 generated spam emails
├── ham_emails.txt          # 2300 generated ham emails
└── README.md
---

## Requirements
- Python 3.10+
- Google Colab (recommended)
- Libraries used: `random`, `math`, `collections`, `numpy`, `matplotlib`

---

## How to Run
1. Open `part1_dataset.ipynb` in Google Colab
2. Run all cells from top to bottom in order:
   - **Part I** — generates dictionaries and synthetic emails
   - **Part II** — trains and tests the Bayesian classifier (MAP + ML)
   - **Part III** — trains and tests the single-neuron neural network

---

## Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `ps` | 0.75 | P(spam word \| spam email) |
| `qs` | 0.20 | P(spam word \| ham email) |
| `l_min` | 5 | Minimum email length (words) |
| `l_max` | 15 | Maximum email length (words) |
| N spam | 1000 | Training spam emails |
| N ham | 2300 | Training ham emails |

---

## Results Summary
| Classifier | Method | Accuracy on 10 test emails |
|------------|--------|---------------------------|
| Bayesian | MAP | 10/10 = 100% |
| Bayesian | ML | 10/10 = 100% |
| Neural Net | Threshold on x_ham | 10/10 = 100% |

---

## Key Design Decisions
- **Laplace smoothing** prevents zero-probability issues for unseen words in the Bayesian model.
- **Log-space arithmetic** avoids floating-point underflow when multiplying many small probabilities.
- **Batch gradient descent** updates weights once per epoch using the full training set.
- The NN threshold `x_T` is derived analytically from the condition `y(1) − y(x_T) = y(x_T) − y(0)`.
