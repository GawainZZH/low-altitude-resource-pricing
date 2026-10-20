# Regulatory Resource Pricing in Low-Altitude Service Markets

Research code and computational data accompanying **Regulatory Resource Pricing in Low-Altitude Service Markets: Platform Competition and Welfare Effects**.

The study examines regulatory resource charges, platform fare competition, access waiting and welfare. The datasets describe parameterized operating states and analytical examples; they are not observations of operating low-altitude markets.

## Installation

Python 3.13, NumPy 2.5.0 and SciPy 1.18.0 were used for the distribution checks. No plotting software or LaTeX installation is needed.

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux), then run:

```bash
python -m pip install -r requirements.txt
```

Download `data.zip` from the [version 1.0.0 release](https://github.com/GawainZZH/low-altitude-resource-pricing/releases/tag/v1.0.0) and put it in the repository root. The private repository requires an authorized GitHub account to access its release assets. Then run:

```bash
python reproduce.py prepare
python reproduce.py verify
```

The application library is available under `experiments/aam_pricing/`. An identical code snapshot is supplied in `code.zip`. The data archive is stored as a release attachment rather than in the code history. `prepare` verifies archive integrity and extracts any missing application files and results without overwriting local modifications. Keep `code.zip`, `data.zip` and `release_manifest.json` together. The preparation command also accepts the equivalent numbered data volumes when obtained as a multipart distribution.

## Contents

| Location after preparation | Contents |
| --- | --- |
| `experiments/aam_pricing/` | Choice, access, platform pricing, welfare, interval bounds and policy calculations |
| `scripts/run_review18_capped_exact.py` | Verification of exact boundary Nash witnesses |
| `results/` | Input designs, accepted responses, unsuccessful candidates, bounds, comparisons and summaries |
| `release_manifest.json` | File sizes and SHA-256 checksums for the distributed evidence |

The identifiers in the application library and saved result paths remain stable so that existing numerical references can be resolved. The repository does not contain the manuscript, submission documents, figure assets or plotting scripts.

## Verification

`python reproduce.py verify` checks all distributed file checksums and recalculates:

- The 24 supplied finite welfare paths and associated mathematical identities.
- Implementation-period welfare over the two supplied fixed-fare, six-charge menus and six evaluation durations.
- Transfer-neutral accounts and endpoint decompositions for the matched fluid and stochastic queue comparisons.
- The exact two-platform equilibrium-selection example.
- The supplied forecast-error comparisons, including incomplete menus and negative gains.
- The independent two-resource example and saved unilateral deviation curves.
- Five exact capacity-boundary Nash witnesses.
- Fixed-fare accounts for the complete 44-response fare-ceiling and participation comparison.

These checks do not solve new platform games. They do recompute fixed-fare behavioral closures, scalar workload integrations and outward interval arithmetic where required. A successful check is not a claim of equilibrium uniqueness or correspondence completeness in every heterogeneous numerical market.

## Recalculation

The following commands recalculate the specified components from the supplied inputs and write separate output files:

```bash
python reproduce.py recompute transition --output outputs/transition.json
python reproduce.py recompute queue-decomposition --output outputs/queue_decomposition.json
python reproduce.py recompute selection --output outputs/selection.json
```

Existing evidence is never overwritten by these commands. The full application library includes the underlying choice, response, platform optimization and finite-menu routines. Re-estimating every platform game is substantially more expensive than checking saved results and is not part of the default verification command.

## Accounting and interpretation

Total surplus credits provider receipts and regulatory receipts as transfers. The user-and-system score is a separate supplementary criterion. Each service institution uses its own zero-charge baseline. Implementation-period calculations hold stationary fare quotations fixed after a state change, with a common initial backlog; they are not transient Nash equilibria.

Saved records distinguish accepted responses, failed candidates, conditional bounds and incomplete menus. These distinctions should be preserved when conducting additional analysis. Monetary and time units, model primitives and solver tolerances are supplied with the corresponding experiment inputs.
