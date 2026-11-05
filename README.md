# Quantitative Quantum Physics & Classical Shadow Entropy Calculations

A collection of Jupyter Notebooks for simulating quantum dynamics of a initial state where all the qubits in spin down state, compressing Trotterized time-evolution using ** Approximate Quantum Compiler(AQC)** tensor network techniques, generating measurement bitstrings  via 'quimb_circuit' and estimating subsystem **Rényi-2 Entanglement Entropy** using **Classical Shadows**.

---

## 📋 Overview

This repository provides simulation pipelines and entropy calculation tools for studying time evolution and entanglement in multi-qubit systems ($L = 104$ qubits). The framework applies:

* **Matrix Product States (MPS)** via `quimb` and `jax` for large-scale density matrix simulations.
* **AQC Block Compression** via `qiskit_addon_aqc_tensor` to compress multiple Trotter time steps into compact, high-fidelity ansatz blocks (placed at $t = 0.6$, $1.2$, or $1.8$).
* **Classical Shadow Tomography** using randomized 3-qubit Pauli string measurements ($3^3 = 27$ basis combinations, 200 shots per operator) to efficiently estimate Rényi-2 entropy from bitstrings without full state tomography.

---

## 📁 Repository Structure & File Descriptions

### 1. `Application_AQC_Entropy_cal_QUIBM_MPS_quimb_shadow.ipynb`
* **Purpose:** Benchmarking and central comparative analysis notebook.
* **System Setup:** System size of $L = 104$ qubits. Computes the Rényi-2 entanglement entropy $S_A^{(2)}$ for a subsystem of size $N_A = 3$ qubits placed at the center of the chain (qubits $q_A = 51, q_B = 52, q_C = 53$).
* **Comparative Scope:**
  1. **Standard `quimb` MPS:** Time evolution without AQC blocks.
  2. **`quimb` MPS + Single AQC Block:** AQC block insertion at $t = 1.8$.
  3. **`quimb` MPS + Multi AQC Blocks:** AQC block insertions at $t = 0.6, 1.2, 1.8$.
  4. **`quimb` Circuits (No AQC):** Quantum circuit dynamics without AQC compression.
  5. **`quimb` Circuits + AQC (t = 1.8):** Circuit dynamics with AQC block at $t = 1.8$.
  6. **`quimb` Circuits + AQC (t = 0.6, 1.4, 1.8):** Circuit dynamics with AQC blocks at $t = 0.6, 1.4, 1.8$.
* **Classical Shadows:** The last three circuit methods incorporate Classical Shadow reconstruction using $200$ shots per Pauli string generated from the $3^3 = 27$ three-qubit Pauli operators ($X, Y, Z$).

---

### 2. Bitstring Sampling Simulators (MPS / Qiskit Circuit Sampling)

* **`Entropy_SIMULATOR_new_ZXZ_op_QUIMB_crc_AQC_NA3.ipynb`**
  * Generates and exports measurement bitstrings for Rényi-2 entropy determination.
  * Incorporates **multi-step AQC compression blocks** placed at $t = 0.6$, $1.2$, and $1.8$ using `quimb` circuits.

* **`Entropy_SIMULATOR_new_ZXZ_op_QUIMB_crc_AQC_NA3_V2.ipynb`**
  * Generates and exports measurement bitstrings for Rényi-2 entropy determination.
  * Incorporates a **single AQC compression block** placed at $t = 1.8$ using `quimb` circuits.

* **`E_Entropy_SIMULATOR_new_ZXZ_op_QUIMB_crc.ipynb`**
  * Generates measurement bitstrings for Rényi-2 entropy determination across time steps.
  * Simulates standard Trotter time evolution **without** AQC block compression using `quimb` circuits.

---

### 3. Classical Shadow Entropy Calculators

* **`Renyi2_entropy_cal.ipynb`**
  * Reconstructs $S_A^{(2)}$ Rényi-2 entropy from classical shadow measurement bitstrings.
  * Designed for bitstring dataset sampled **without AQC blocks** (uncompressed Trotterized evolution).

* **`Renyi2_entropy_cal_AQC.ipynb`**
  * Reconstructs $S_A^{(2)}$ Rényi-2 entropy from classical shadow measurement bitstrings.
  * Tailored for bitstring datasets sampled from **AQC-compressed circuits**.

---

## ⚙️ Key Methodologies

### 1. Quantum System & Subsystem Definition
* **Total System Size ($L$):** 104 qubits
* **Subsystem Size ($N_A$):** 3 qubits at the center chain ($L/2 - 1, L/2, L/2 + 1$)
* **Bond Dimension ($M$):** Max bond dimension $\chi = 128$, truncation cutoff $\epsilon = 10^{-10}$.

### 2. AQC Tensor Compression
Trotterized evolution time steps ($\Delta t = 0.3$) are compressed into parameterized ansatz blocks by maximizing state fidelity using `qiskit_addon_aqc_tensor` combined with JAX-backed autodiff tensor network simulations.

### 3. Classical Shadow Tomography
* **Pauli Basis Set:** $3^3 = 27$ combinations of three-qubit Pauli operators ($X, Y, Z$).
* **Shots:** 200 shot samples taken per Pauli string operator.
* **Density Matrix Estimation:** Constructs snapshot state representations $\rho^{(k)}$ to compute $S_2(\rho_A) = -\ln \text{Tr}(\rho_A^2)$.

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.9+ is installed along with the following dependencies:

```bash
pip install numpy scipy qiskit qiskit-aer quimb jax jaxlib
