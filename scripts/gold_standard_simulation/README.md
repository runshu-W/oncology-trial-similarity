# Gold-standard borrowability simulation

A simulation study in which the correct borrowing decision is known by
construction, used to evaluate the borrowing layer of the pipeline under known
truth. Structured following ADEMP (Morris, White & Crowther 2019).

Pure NumPy — no PyTorch, no SciPy. The reverse-mode autodiff needed by the
two-head model is implemented in `autodiff.py` and gradient-checked against
central finite differences.

## Why this exists

The retrospective analyses on real registry text show the pipeline *runs*, but
cannot show whether it borrows from the *right* trials, because the truth is
unknown. This simulation supplies the missing counterfactual.

## Layout

| File | Purpose |
|---|---|
| `dgm.py` | Data-generating mechanism; NSCLC-anchored, non-linear, deliberately not the fitted model. Also the external-control DGM. |
| `methods.py` | All priors compared: rule mixture, two-head DeepSets variants, robust-MAP, power prior, UIP (Dirichlet and JS), pooling, SAM adapter. |
| `metrics.py` | Operating characteristics, borrowing-decision diagnostic accuracy, oracle-allocation agreement, learned-ESS response. All with Monte Carlo standard errors. |
| `autodiff.py` | Minimal reverse-mode autodiff on NumPy arrays. |
| `run_simulation.py` | Driver for the estimation scenarios and the design worlds. |
| `run_external_control.py` | Driver for the hybrid-control (external control arm) scenario. |
| `build_outputs.py` | Tables and figures. |
| `test_simulation.py` | 30 correctness checks, including a leakage test. |
| `test_autodiff.py` | Analytic-versus-numeric gradient checks. |

## Design decisions that matter

**The data-generating mechanism is not the analysis model.** True response rates
come from latent clinical attributes through a non-linear map with interaction
terms; methods only ever see noisy nine-dimensional feature scores. A simulation
whose DGM coincides with the fitted model is self-confirming.

**Surface similarity is decoupled from borrowability by construction.** Textual
similarity is driven by disease and regimen; the true rate is driven mainly by
line of therapy and PD-L1 status. Trap donors (similar-looking, not
exchangeable) and hidden gems (dissimilar-looking, exchangeable) therefore arise
from clinical structure, not from artificial injection.

**The gold standard is parameter-level, not surface-level.** A donor is
borrowable iff its *observed* rate lies within `epsilon = 0.05` of the query's
true rate. Observed rather than latent, because a donor whose endpoint
definition biases its reported rate will damage a borrowed analysis regardless
of its underlying biology.

**Type I error needs a null that does not depend on the realised truth.** The
simulation is embedded in a single-arm design testing `H0: theta <= 0.20` at the
one-sided 0.025 level, evaluated in a null world (`theta_0 = 0.20`, rejection
rate *is* the type I error) and an alternative world (`theta_0 = 0.35`, rejection
rate *is* the power). `theta_0` is pinned by shifting the query *and every donor*
by a common logit offset, so the exchangeability structure of the donor pool is
preserved and only the location moves.

Note that a Bayesian test of a point null with a flat prior does not attain its
nominal level exactly at these sample sizes, because the binomial is discrete.
The no-borrowing rejection rate is reported as the achievable reference level
alongside the nominal 0.025.

**Prospective versus retrospective methods are not compared as equals.** The
set-level conflict signal (`prospective_conflict` in `methods.py`) uses only the
donors' already-published outcomes, never the query's held-out outcome, and is
therefore available at design time. SAM and UIP-JS both compute their conflict
measure against the current trial's data and are labelled as retrospective
wherever they appear.

**Scenario S1 is included to be favourable to the pooled competitors.** Nearly
all donors are exchangeable there, which is the setting full pooling is built
for. Reporting only adversarial scenarios would be cherry-picking.

## Reproducing

```bash
cd scripts/gold_standard_simulation

# 1. Train the three two-head variants (~1 min)
python3 run_simulation.py --mode train --train-size 300 --epochs 60

# 2. Estimation scenarios S1-S5
python3 run_simulation.py --mode scenarios --replicates 2000

# 3. Design worlds: type I error and power under conflict
python3 run_simulation.py --mode design --design-replicates 2000

# 4. External control arm
python3 run_external_control.py --replicates 2000

# 5. Tables and figures
python3 build_outputs.py

# Tests
python3 test_simulation.py && python3 test_autodiff.py
```

Every scenario and design cell is checkpointed to its own part file, so an
interrupted run resumes rather than restarting. `--cell` runs a single design
cell, which allows several to be run in parallel:

```bash
python3 run_simulation.py --mode design --cell null_0.5 --no-combine
```

Results land in `artifacts/gold_standard_simulation/`. The summary CSVs, tables
and figures are tracked in git; per-cell checkpoints and trained weights are not.

Runtime is roughly 17 ms per replicate across all 13 methods, so a 2000-replicate
cell takes about 35 seconds. The full study is a few minutes.

## Reproducibility

Scenario seeds are derived with `zlib.crc32`, not Python's built-in `hash`, which
is randomised per process unless `PYTHONHASHSEED` is set. `run_config.json`
records the replicate counts, seed, design constants and conflict grid for each
run.

## A note on the UIP comparator

`uip_dirichlet` and `uip_js` implement Jin & Yin (2021, *Statistics in Medicine*
40(25):5657–5672), section 3.2. The amount parameter `M` is marginalised on a
grid rather than by MCMC, which turns the UIP into a finite beta mixture and lets
it be scored on the same footing as every other prior here.

One deliberate deviation from the paper is documented in the code: the unit
information `1/(theta(1-theta))` is evaluated at the Jeffreys posterior mean
`(y+0.5)/(n+1)` rather than at the raw MLE. With the small arms typical of
oncology, `y_k = 0` is common and the raw MLE sends the unit information to
infinity, letting a single zero-response donor dominate the total information and
inflating the borrowed ESS by an order of magnitude. The Jeffreys prior is the
same one the paper already uses to construct the UIP-JS weights, so this is
internally consistent with the method rather than an ad hoc patch, and it
restores the paper's own stated property that the borrowed ESS is approximately
`M - 1`.

## Related scripts

- `../fix_endpoint_units.py` — unit-aware conversion of ClinicalTrials.gov
  outcome rows into rates. Dispatches on the reported unit and refuses to convert
  rows whose unit is missing or unrecognised, rather than guessing.
- `../audit_orr_units.py` — quantifies the impact of the endpoint-unit defect on
  the corpus and on the borrowing dataset; writes the per-query correction table.
