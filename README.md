
# Scheduler-Aware CPU Burst-Time Prediction

### Evaluating the Impact of Burst-Time Prediction Errors on CPU Scheduling

An Operating Systems research project investigating how CPU burst-time prediction accuracy influences scheduling decisions in **Shortest Job First (SJF)** and **Shortest Remaining Time First (SRTF)** algorithms.

The project explores whether traditional prediction accuracy metrics such as MAE and RMSE adequately represent scheduling performance, and proposes a scheduler-aware metric called **Scheduling Decision Error Rate (SDER)**.

**Course:** BCSE303P – Operating Systems  
**Institution:** VIT Vellore  
**Project Status:** Review 2 – Partial Implementation

---

## 1. Problem Statement

CPU scheduling algorithms such as SJF and SRTF rely on estimates of future CPU burst times.

However, actual CPU burst durations are generally unknown before execution.

Traditional prediction models are evaluated using numerical accuracy metrics such as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

These metrics measure prediction accuracy but do not directly capture whether prediction errors change the relative execution order of processes.

This project investigates how prediction-induced ordering errors influence CPU scheduling performance.

## 2. Research Objectives

1. Implement traditional CPU burst-time prediction techniques.
2. Implement machine-learning-based burst-time prediction.
3. Simulate SJF and SRTF scheduling using predicted burst times.
4. Evaluate prediction accuracy using MAE and RMSE.
5. Measure scheduling performance using waiting time, turnaround time, and response time.
6. Develop a Scheduling Decision Error Rate (SDER) metric.
7. Investigate whether lower prediction error necessarily produces better scheduling outcomes.
8. Validate findings using real-world workload traces.

## 3. Proposed System Architecture

```text
          Workload Dataset
                 |
                 v
          Data Preprocessing
                 |
                 v
        CPU Burst Prediction
           /           \
          /             \
 Exponential         Machine Learning
  Averaging          Linear Regression
          \             /
           \           /
                 |
                 v
         CPU Scheduling
           /        \
          SJF       SRTF
                 |
                 v
       Performance Evaluation
                 |
                 v
     MAE | RMSE | Waiting Time
      Turnaround Time | SDER
```

## 4. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core implementation |
| NumPy | Numerical operations |
| Pandas | Dataset processing |
| Scikit-learn | Machine learning and evaluation |
| Matplotlib | Experimental visualizations |
| Git & GitHub | Version control |
| VS Code | Development environment |

## 5. Current Implementation

The following modules have been implemented as part of the Review 2 prototype.

### Module 1: Synthetic Dataset Generation

Generates 1,000 simulated CPU processes containing:

- Process ID
- Arrival time
- Historical CPU burst
- Actual CPU burst

File: `src/generate_data.py`

Output: `data/synthetic_workload.csv`

### Module 2: Data Preprocessing

Validates the generated dataset by checking:

- Missing values
- Duplicate process IDs
- Invalid CPU burst durations
- Negative arrival times
- Chronological ordering

File: `src/preprocess.py`

Output: `data/processed_workload.csv`

### Module 3: CPU Burst-Time Prediction

Implements two prediction approaches.

**Exponential Averaging (Simplified Baseline)**

Uses a one-step estimate based on the historical burst and a fixed initial prediction.

The traditional exponential averaging formula is:

τ(n+1) = α × t(n) + (1 − α) × τ(n)

**Linear Regression**

Trains a regression model using historical CPU burst information to predict actual burst durations.

The dataset is divided chronologically into:

- 80% training data
- 20% testing data

File: `src/predict.py`

Output: `results/predictions.csv`

### Module 4: SJF Scheduling Simulation

Implements non-preemptive Shortest Job First scheduling.

Three scheduling scenarios are evaluated:

1. Oracle SJF using actual burst times.
2. SJF using exponential averaging predictions.
3. SJF using linear regression predictions.

The simulator calculates:

- Process execution order
- Start and completion times
- Waiting time
- Turnaround time
- Response time

File: `src/sjf.py`

Output: `results/sjf_results.csv`

### Module 5: SJF Unit Testing

A manually verifiable test case validates the basic SJF scheduling order and timing calculations.

File: `src/test_sjf.py`

Result: **Test Passed**

---

## 6. Preliminary Experimental Results

### Prediction Accuracy

| Metric | Exponential Averaging | Linear Regression |
|--------|----------------------:|------------------:|
| MAE | 3.8546 | 3.0005 |
| RMSE | 4.7197 | 3.6869 |

Linear regression achieved lower prediction error on the synthetic test dataset.

### SJF Scheduling Performance

| Metric | Oracle SJF | Exponential Averaging | Linear Regression |
|--------|-----------:|----------------------:|------------------:|
| Average Waiting Time | 724.395 | 777.085 | 777.085 |
| Average Turnaround Time | 736.335 | 789.025 | 789.025 |
| Average Response Time | 724.395 | 777.085 | 777.085 |

### Preliminary Observation

Despite achieving better numerical prediction accuracy, linear regression produced the same aggregate SJF scheduling results as exponential averaging.

This illustrates why prediction accuracy and scheduling usefulness should be evaluated separately.

These results are preliminary and based on synthetic data. They do not establish general performance on real-world workloads.

---

## 7. Project Structure

```text
CPU-Burst-Prediction/
|
|-- data/
|   |-- synthetic_workload.csv
|   |-- processed_workload.csv
|
|-- src/
|   |-- generate_data.py
|   |-- preprocess.py
|   |-- predict.py
|   |-- sjf.py
|   |-- test_sjf.py
|
|-- results/
|   |-- predictions.csv
|   |-- sjf_results.csv
|
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## 8. Installation and Execution

### Clone the repository

```bash
git clone https://github.com/SAMS50/CPU-Burst-Prediction.git
cd CPU-Burst-Prediction
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the modules

Run the following commands from the project root, in order:

```bash
python src/generate_data.py
python src/preprocess.py
python src/predict.py
python src/sjf.py
python src/test_sjf.py
```

## 9. Proposed Scheduling Decision Error Rate (SDER)

The project proposes SDER to quantify prediction-induced process ordering errors.

For eligible process pairs:

**SDER = (Incorrectly ordered pairs / Total eligible pairs) × 100**

A lower SDER indicates fewer incorrect pairwise ordering decisions.

The final definition will address process arrival times, ties, eligible pairs, and preemptive SRTF decision points.

SDER has not yet been implemented or experimentally validated.

## 10. Future Work – Review 3

The remaining research work includes:

- Integration and validation of real-world workload traces.
- Implementation of the SRTF scheduling simulator.
- Formal definition and implementation of SDER.
- Additional predictor comparisons where appropriate.
- Comprehensive scheduling performance evaluation.
- Analysis of the relationship between MAE, RMSE, SDER, and scheduling outcomes.
- Final experimental visualizations and research conclusions.

## 11. Current Limitations

- Initial experiments use synthetic workloads.
- The exponential averaging baseline uses a simplified one-step estimate.
- Only non-preemptive SJF scheduling is currently implemented.
- SDER and SRTF remain future implementation tasks.
- Additional scheduling test cases and real-world validation are required.

## 12. Research Contribution

The intended contribution is a scheduler-aware evaluation framework that connects CPU burst-time prediction errors with process ordering decisions and downstream scheduling performance.

The project aims to determine whether pairwise scheduling decision errors provide useful explanatory information beyond conventional regression metrics.

---

**Author:** Saamiya Siddique  

