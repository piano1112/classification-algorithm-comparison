# Classification Algorithm Comparison

Custom implementations of **Naive Bayes** and **Decision Tree** classifiers built from scratch in Python — no scikit-learn, no libraries. Benchmarked on the Pima Indians Diabetes dataset to predict diabetes onset from clinical measurements.

Built as part of COMP3608 (Introduction to AI (Advanced)) at the University of Sydney.

## Algorithms

### Naive Bayes (`naive_bayes.py`)

Gaussian Naive Bayes classifier for continuous attributes. Computes class-conditional probability density using the normal distribution, then applies Bayes' theorem to classify each instance. Ties default to the positive class.

### Decision Tree (`decision_tree.py`)

ID3-style decision tree using **information gain** (entropy-based) for attribute selection. Handles nominal attributes with multi-way splits. Falls back to the majority class for unseen attribute values.

### Decision Tree* (`decision_tree*.py`)

An enhanced variant with **pre-pruning**: splits are blocked when a node has fewer than 77 training examples (tuned via cross-validation on the dataset). This reduces overfitting and improved generalisation accuracy on the Pima dataset compared to the unpruned tree.

## Dataset

The [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) contains 768 instances with 8 clinical attributes:

| Attribute | Description |
|-----------|-------------|
| `tp` | Times pregnant |
| `gc` | Glucose concentration |
| `bp` | Blood pressure |
| `sft` | Skin fold thickness |
| `si` | Serum insulin |
| `bmi` | Body mass index |
| `dpf` | Diabetes pedigree function |
| `age` | Age |
| `class` | Outcome (`yes` / `no`) |

Source: National Institute of Diabetes and Digestive and Kidney Diseases, modified for COMP3608.

## Project Structure

```
naive_bayes.py                       → Gaussian Naive Bayes classifier
decision_tree.py                     → Standard ID3 decision tree (MyDT)
decision_tree*.py                    → Pre-pruned decision tree (MyDT*)
report_COMP3608_A2.pdf               → Analysis report with results
data/
  pima-indians-diabetes.csv          → Full dataset (continuous attributes)
  pima-indians-diabetes-discrete.csv → Discretised version for decision trees
  pima-indians-diabetes-info.txt     → Dataset documentation
  pima-folds.csv                     → 10-fold cross-validation splits
  pima.csv                           → Normalised version of the dataset
samples/
  train.csv                          → Small sample training set (weather data)
  test.csv                           → Small sample test set
```

## Usage

```bash
python3 -c "
from naive_bayes import classify_nb
from decision_tree import classify_dt

# Returns list of predicted classes
nb_results = classify_nb('data/pima-indians-diabetes.csv', 'data/pima-indians-diabetes.csv')
dt_results = classify_dt('data/pima-indians-diabetes-discrete.csv', 'data/pima-indians-diabetes-discrete.csv')
"
```

## Key Takeaway

The pre-pruned Decision Tree* (MyDT*) achieved the highest accuracy on nominal data across all tested classifiers — including the unpruned tree and Naive Bayes — by preventing overfitting through a minimum-sample threshold tuned via cross-validation. Full analysis is in the [report](report_COMP3608_A2.pdf).
