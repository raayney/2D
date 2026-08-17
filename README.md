# 2D Nuclear Chain Reaction Simulator

A 2D probabilistic Monte Carlo simulation and interactive Streamlit web dashboard for modeling nuclear neutron chain reactions, criticality regimes, and multi-channel heat deposition within a discrete Uranium-235 (235U) core geometry.

This project was developed at Sorbonne University as part of the UL1SXARE curriculum by Rayney Poon Shao Rui, Boubacar, and Yasser.

---

## Overview

This repository implements a 2D lattice model that simulates the stochastic transport and interaction dynamics of neutrons within a nuclear reactor core. Beyond tracking neutron populations and calculating empirical criticality (k_eff), the model tracks total energy deposition and spatial thermal distribution across three particle decay channels:

* **Prompt Neutrons:** Emitted via Poisson distribution (lambda = 2.43) with kinetic energies sampled from the Watt fission spectrum.
* **Fission Fragments:** Heavy ions carrying the majority of fission kinetic energy (~168 MeV) deposited locally over a short range (1-2 cells).
* **Beta Electrons (beta-):** Produced via fission fragment decay (~6 electrons, 8 MeV total) distributed using a Dirichlet prior, traveling longer ranges (5-10+ cells) and driving wall energy leakage.

---

## Features

* **Interactive Streamlit Web Dashboard:** Customize physical parameters (grid size, fuel fraction, interaction probabilities, range boundaries, Watt spectrum coefficients) and execute Monte Carlo runs in real time.
* **Criticality Regime Classification:** Automatically categorizes reactor runs into Extinction (k_eff < 1), Quasi-critical (k_eff ~ 1), or Growth (k_eff > 1).
* **Dynamic Visualization:** Generates real-time population plots, core heatmaps, boundary wall thermal distribution bar charts, and an animated GIF tracking spatial particle evolution.
* **Energy Balance & Efficiency Estimation:** Breaks down core vs. wall heat absorption across all particle species and calculates estimated electrical energy output (eta_collectible = 0.94, eta_thermal = 0.33).

---

## Installation & Setup

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/nuclear-chain-reaction-2d.git
cd nuclear-chain-reaction-2d
```

### 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install streamlit numpy matplotlib pandas pillow plotly
```

---

## Running the Application

Launch the Streamlit dashboard by running:

```bash
streamlit run app.py
```

Once started, open your web browser and navigate to `http://localhost:8501`. Adjust the control sliders in the sidebar and click **Simuler** to trigger the simulation.

---

## Key Physical Parameters

| Parameter | Default Value | Physical Description |
| :--- | :--- | :--- |
| `taille` | 50 | Reactor grid dimension (N x N cells) |
| `fraction_u` | 0.80 | Fraction of grid cells containing 235U fuel |
| `p_fiss` | 0.10 | Probability of fission upon 235U interaction |
| `p_abs` | 0.30 | Probability of radiative capture/absorption |
| `p_diff` | 0.60 | Implicit scattering probability (1 - p_fiss - p_abs) |
| `k` | 2.43 | Mean prompt neutrons emitted per fission (Poisson) |
| `a`, `b` | 0.988 MeV, 2.249 MeV^-1 | Watt spectrum energy distribution parameters |
| `TKE_moy`, `TKE_sig` | 168 MeV, 8 MeV | Mean and std. dev. of total kinetic energy from fragments |
| `fp_min`, `fp_max` | 1, 2 cells | Interaction range for fission fragments |
| `beta_E` | 8.0 MeV | Total beta electron energy release per fission event |
| `bp_min`, `bp_max` | 5, 10 cells | Interaction range for beta particles |

---

## Academic Context & References

This implementation accompanies the research report *"Simulation d'une Réaction en Chaîne Nucléaire 2D: Un Modèle Spatial des Réactions en Chaîne de Neutrons et de la Criticité"*.

Key references supporting the model parameters:
1. **Watt Spectrum:** Griffin, J.J. (1999) & Sardet, A. (2015) — Parameters a = 0.988 MeV, b = 2.249 MeV^-1.
2. **Neutron Multiplicity:** IAEA EXFOR / Nuclear Data Sheets — Mean yield nu = 2.43.
3. **Fission Fragment Energy:** NASA TM-2016-004957 / LANL CGMF — Average TKE = 168 MeV, sigma = 8 MeV.
4. **Beta Decay & Stopping Power:** SRIM (Ziegler et al., 2010) — Range differential between heavy ions (1-2 cells) and electrons (5-10 cells).
