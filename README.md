# ICL-Capstone-BBO
Imperial College ML &amp; AI capstone: Bayesian optimisation of eight black-box functions, with reproducible Python notebooks, query results and reflections.

# Black-Box Optimisation Capstone

**Imperial College London | Professional Certificate in Machine Learning and Artificial Intelligence**

An ongoing capstone project exploring how Bayesian optimisation can identify high-performing inputs for eight unknown functions under a limited evaluation budget.

## Project overview

In a black-box optimisation problem, the relationship between inputs and outputs is unknown. We can observe the result of an evaluation, but cannot inspect the underlying function or calculate its gradients directly.

This project starts with a small dataset of input-output observations for each of eight functions. The objective is to select new inputs that maximise each function's output. Each submission round proposes one new input vector per function, with returned evaluations informing subsequent decisions.

The challenge is to make effective use of limited evaluations by balancing:

- **Exploration:** sampling regions where the function is poorly understood.
- **Exploitation:** sampling regions expected to produce high outputs.

This repository documents the code, reasoning and evidence behind those decisions.

## Project goals

- Improve the best observed output for each of the eight functions.
- Apply Bayesian optimisation to problems with different dimensionalities and response patterns.
- Balance predicted performance against uncertainty when selecting queries.
- Adapt the strategy as new observations become available.
- Maintain clear, reproducible notebooks and a traceable record of submissions.
- Evaluate the strengths and limitations of the approach.

## The optimisation problems

Each function accepts a numerical input vector and returns a single numerical output.

| Function | Input dimensions | Initial observations | Objective |
|---|---:|---:|---|
| Function 1 | 2 | 10 | Maximise output |
| Function 2 | 2 | 10 | Maximise output |
| Function 3 | 3 | 15 | Maximise output |
| Function 4 | 4 | 30 | Maximise output |
| Function 5 | 4 | 20 | Maximise output |
| Function 6 | 5 | 20 | Maximise output |
| Function 7 | 6 | 30 | Maximise output |
| Function 8 | 8 | 40 | Maximise output |

The initial datasets contain **175 observations** in total. Higher-dimensional problems require searching a larger space with relatively little data.

The current notebook generates submission coordinates between `0.000000` and `0.999999`, formatted to six decimal places and separated by hyphens. For example, a two-dimensional query is:

```text
0.123456-0.654321
```

## Optimisation approach

### Gaussian process surrogate models

For Functions 2–8, Gaussian process models estimate both the expected output and predictive uncertainty at untested inputs. These surrogate models provide a relatively inexpensive way to assess candidate points before requesting a genuine black-box evaluation.

The week 1 implementation uses a Matérn-based kernel and standardises the observed outputs during model fitting.

### Upper Confidence Bound acquisition

The initial query selection strategy uses an Upper Confidence Bound (UCB) acquisition function:

**UCB(x) = predicted mean(x) + 2 × predicted standard deviation(x)**

Candidates can therefore be attractive because they have a high predicted output, substantial uncertainty or a combination of both. The uncertainty weight is initially set to 2 and can be reconsidered as evidence accumulates.

Candidate generation and local numerical refinement are used to search for promising acquisition values. This process does not guarantee a global optimum.

### Function 1: space-filling exploration

Function 1's initial outputs are predominantly close to zero, providing limited evidence about the location of positive peaks.

Its week 1 candidate therefore uses space-filling exploration. A Gaussian process is fitted for diagnostic purposes, but does not select this function's initial query.

### Sequential learning

Each round follows the same broad workflow:

1. Validate the available input-output observations.
2. Fit the surrogate models and review their diagnostics.
3. Select one candidate input vector per function.
4. Record the query and explain its selection.
5. Submit queries through the competition portal.
6. Record the actual submitted inputs and returned outputs.
7. Update the observations and reassess the strategy.

The notebooks generate proposed queries. They do not automatically submit them or evaluate the hidden functions.

## Key technologies

| Technology | Purpose |
|---|---|
| Python | Core implementation and analysis |
| Jupyter Notebook | Executable code, explanations and results |
| NumPy | Numerical arrays and observation data |
| SciPy | Candidate sampling and numerical optimisation |
| scikit-learn | Gaussian process regression and kernel modelling |
| Matplotlib | Diagnostic plots and progress visualisation |
| CSV and JSON | Query exports and structured experiment records |
| Git and GitHub | Version control and project documentation |

## Repository organisation

The intended structure as the project develops is:

| Location | Purpose |
|---|---|
| `README.md` | Project overview and running instructions |
| `notebooks/` | A separately documented notebook for each week |
| `data/` | Initial observations and subsequent evaluation data |
| `submissions/` | Submitted queries and accompanying reflections |
| `results/` | Exported records, plots and progress summaries |
| `docs/` | Supporting methodology, limitations and references |
| `requirements.txt` | Python dependencies for reproducibility |

Folders and supporting files will be added as the project develops. The week 1 notebook embeds the original initial datasets and does not require a separate `data/` folder.

## Getting started

### Set up the environment

Use Python 3.12 for consistency with the documented week 1 environment. From a local copy of the repository, create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

The activation command above is for macOS and Linux. On Windows PowerShell, use `.venv\Scripts\Activate.ps1`.

Install the versions recorded in the week 1 notebook:

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 matplotlib==3.10.8 jupyter
```

### Run the notebook

1. Start Jupyter with `jupyter notebook`.
2. Open `BBO_week1_2026-11-08.ipynb` wherever it is stored in the repository.
3. Run the cells in order.
4. Review the diagnostics, proposed queries and exported records.

The notebook writes generated files to `week1_outputs` relative to its working directory. Embedded input files are checked using SHA-256 hashes before loading.

## Evaluation and reproducibility

Progress will be assessed using the **best actual observed output per function** and its improvement over the initial baseline.

Model predictions and uncertainty estimates support query selection, but are not evidence of achieved improvement. Returned competition evaluations are required to measure progress.

Reproducibility is supported by fixed random seeds, recorded package versions, input validation and structured exports. Small numerical differences may occur between platforms.

Each week should retain its proposed queries, actual submitted inputs, returned outputs and selection rationale. Previous weeks should remain available so that changes in strategy can be reviewed.

## Status and limitations

The week 1 notebook documents the initial-data analysis and proposes eight queries. Its recorded results do not yet include returned evaluations for those queries.

The main limitations are sparse observations, increasing search difficulty in higher dimensions and sensitivity to surrogate modelling assumptions. Predictive uncertainty may be unreliable where those assumptions do not fit the hidden function.

The project aims to identify progressively better inputs. It does not claim to recover the exact functions or establish their global maxima.

## Practical relevance

Black-box optimisation is relevant to hyperparameter tuning, experimental design and other settings where evaluations are expensive.

My broader interests include its potential application to Media and Entertainment and Ad Tech, particularly audience prediction, advertising inventory allocation and TV schedule optimisation. Those applications would require additional treatment of discrete decisions, business constraints and changing conditions.

## References

- [scikit-learn: Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [SciPy documentation](https://docs.scipy.org/doc/scipy/)
- [Jupyter documentation](https://docs.jupyter.org/en/latest/)
