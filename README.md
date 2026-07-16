# Feigenbaum Chaos Explorer

Interactive Tkinter laboratory for period doubling, deterministic chaos, and Feigenbaum universality.

The application does more than draw a logistic-map bifurcation diagram. It numerically estimates the two Feigenbaum scaling constants and compares their convergence across several nonlinear maps:

- Parameter-space scaling: **δ ≈ 4.6692016091**
- State-space scaling: **α ≈ −2.5029078751**

## Features

### Interactive visualizations

- Bifurcation diagram with density-based rendering
- Lyapunov spectrum
- Orbit time series and cobweb diagram
- Feigenbaum delta convergence
- Feigenbaum alpha scaling
- Universality comparison across nonlinear maps
- Zoom, pan, reset, presets, and high-resolution PNG export

### Supported maps

- Logistic map: `xₙ₊₁ = r xₙ(1 − xₙ)`
- Sine map: `xₙ₊₁ = r sin(πxₙ)`
- Quadratic map: `xₙ₊₁ = 1 − axₙ²`

### Numerical methods

- Vectorized NumPy bifurcation rendering
- Burn-in and long-run orbit sampling
- Lyapunov exponent estimation
- Superstable cycle detection
- Bisection root finding
- Period-doubling ratio estimation
- Critical-orbit distance scaling

## Project structure

```text
Feigenbaum-Chaos-Explorer/
├── run.py
├── pyproject.toml
├── requirements.txt
├── src/
│   └── feigenbaum_explorer/
│       ├── app.py
│       ├── maps.py
│       └── numerics.py
├── tests/
│   └── test_numerics.py
└── output/
```

## Installation

```bash
conda create -n feigenbaum-chaos python=3.11 -y
conda activate feigenbaum-chaos
pip install -r requirements.txt
```

Alternatively, install the project in editable mode:

```bash
pip install -e .
```

## Run

```bash
python run.py
```

After editable installation:

```bash
feigenbaum-explorer
```

## Controls

| Action | Control |
|---|---|
| Render selected view | Render button or Enter |
| Zoom and pan | Matplotlib navigation toolbar |
| Apply parameter preset | Preset dropdown |
| Reset application | Reset button |
| Export current view | Save PNG or Ctrl+S |
| Exit | Escape |

## Mathematical background

For a sequence of period-doubling parameter values `rₙ`, the ratio

```text
δₙ = (rₙ₋₁ − rₙ₋₂) / (rₙ − rₙ₋₁)
```

approaches Feigenbaum's delta constant. The corresponding critical-orbit distances `dₙ` satisfy approximately

```text
αₙ = −dₙ₋₁ / dₙ
```

and converge toward Feigenbaum's alpha constant.

The striking result is universality: smooth one-dimensional maps with a quadratic maximum exhibit the same asymptotic scaling even though their formulas and bifurcation locations differ.

## Tests

```bash
pytest -q
```