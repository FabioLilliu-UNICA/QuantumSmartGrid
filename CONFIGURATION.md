# Configuration Guide

This document explains how to configure the input data and register sizes used by the quantum load-shifting optimization script.

## Problem data

The main input data are defined near the beginning of the script.

### `f`

`f` represents the fixed net-load contribution over the considered time horizon.

Example:

```python
f = np.array([1, 0, -2, -2])
```

Its length must be equal to `N`.

---

### `s_list`

`s_list` contains the flexible load profiles.

Example:

```python
s_list = [
    np.array([-1, 4, -4, -1]),
    np.array([-3, 3, -1, -2]),
]
```

Each vector in `s_list` must have length `N`.

The number of flexible loads must satisfy:

```python
L == len(s_list)
```

---

### `p_pos`

`p_pos` contains the buying price associated with each time interval.

Example:

```python
p_pos = np.array([10, 8, 12, 9])
```

Its length must be equal to `N`.

---

### `p_neg`

`p_neg` contains the selling price associated with each time interval.

Example:

```python
p_neg = np.array([4, 6, 3, 5])
```

Its length must be equal to `N`.

---

# Main parameters

## `N`

`N` is the number of time intervals and also the number of possible circular shifts for each flexible load.

For example:

```python
N = 4
```

All vectors in

```python
f
s_list
p_pos
p_neg
```

must have length `N`.

The current implementation assumes that `N` is a power of two.

For example:

```text
N = 2
N = 4
N = 8
N = 16
```

are supported by the current encoding.

The script automatically defines

```python
SHIFT_BITS = int(np.ceil(np.log2(N)))
```

so that each load shift is represented using `SHIFT_BITS` qubits.

---

## `L`

`L` is the number of flexible loads.

It must satisfy:

```python
L = len(s_list)
```

For example, if

```python
s_list = [
    np.array([...]),
    np.array([...]),
    np.array([...]),
]
```

then

```python
L = 3
```

must be used.

---

# Register sizes

The values of `V_BITS` and `C_BITS` must be selected carefully.

Choosing values that are too small may cause arithmetic overflow and produce incorrect results.

Choosing unnecessarily large values increases the number of qubits, circuit depth, and gate count.

## `V_BITS`

`V_BITS` is the number of qubits used to encode the temporary net-energy value.

For a configuration `k`, the corresponding value at time `t` is

```text
v_k(t) = f(t) + sum of the shifted flexible-load contributions
```

The `v` register uses signed two's-complement representation.

Therefore, with `V_BITS = b`, the allowed range is

```text
-2^(b-1)  <=  v_k(t)  <=  2^(b-1) - 1
```

for every configuration and every time interval.

For example, if

```python
V_BITS = 6
```

then the allowed range is

```text
-32 <= v_k(t) <= 31
```

`V_BITS` should therefore be chosen so that every possible net-energy value fits within this range.

Using the minimum safe value is recommended, since reducing `V_BITS` can significantly reduce circuit depth and gate count.

---

## `C_BITS`

`C_BITS` is the number of qubits used to encode the total cost.

The cost register also uses signed arithmetic.

`C_BITS` must be large enough to represent all relevant cost values and, in particular, all values involved in the threshold comparison.

The comparison checks the sign of

```text
C(k) - threshold
```

so all relevant values of this difference must fit in the signed range

```text
-2^(C_BITS-1)
<=
C(k) - threshold
<=
2^(C_BITS-1) - 1
```

without overflow.

As with `V_BITS`, using the minimum safe value is recommended.

Reducing `C_BITS` can substantially reduce the number of gates required by:

- cost accumulation;
- QFT and inverse QFT operations;
- threshold comparison;
- uncomputation.

---

# Numerical values

The reversible quantum arithmetic currently assumes integer-valued quantities.

Therefore, the following should contain integers:

```python
f
s_list
p_pos
p_neg
```

If the original data contain decimal values, they should first be multiplied by a common scaling factor and converted to integers.

For example:

```text
1.25 -> 125
0.80 -> 80
```

using a scaling factor of `100`.

The same scaling convention must be used consistently throughout the problem.

---

# Load shifting

Flexible-load profiles are shifted using:

```python
np.roll(...)
```

Therefore, the shifts are circular.

For example, shifting

```text
[a, b, c, d]
```

by one position produces

```text
[d, a, b, c]
```

Values shifted beyond the end of the time horizon re-enter at the beginning.

---

# Notation

The current source code uses the following notation:

```text
N = number of time intervals / possible shifts
L = number of flexible loads
```

In the associated mathematical formulation and paper, these quantities are denoted respectively by:

```text
T = number of time intervals / possible shifts
M = number of flexible loads
```

Therefore:

```text
code N  <->  paper T
code L  <->  paper M
```

This distinction should be kept in mind when comparing the implementation with the mathematical description.

---

# Total number of qubits

Ignoring additional ancillas introduced during gate decomposition, the main logical circuit uses approximately

```text
L * ceil(log2(N)) + V_BITS + C_BITS + 1
```

qubits.

These correspond to:

```text
configuration register
+ v register
+ cost register
+ comparison flag
```

For example, with

```python
N = 4
L = 2
V_BITS = 4
C_BITS = 8
```

the configuration register requires

```text
2 * log2(4) = 4 qubits
```

and the complete logical circuit uses

```text
4 + 4 + 8 + 1 = 17 qubits
```

before additional decomposition ancillas.

---

# Computational requirements

The complete circuit can be evaluated using exact statevector simulation.

Statevector simulation scales exponentially with the number of qubits.

A circuit containing `Q` qubits requires a statevector with

```text
2^Q
```

complex amplitudes.

Consequently, increasing any of the following may substantially increase simulation time and memory requirements:

```text
N
L
V_BITS
C_BITS
```

Small problem instances are therefore recommended for exact simulation.

---

# Parameters that should be updated together

When defining a new problem instance:

1. Set `N` equal to the length of the time-series data.
2. Set `L` equal to `len(s_list)`.
3. Make sure every vector in `f`, `s_list`, `p_pos`, and `p_neg` has length `N`.
4. Verify that `N` is a power of two.
5. Choose the smallest safe value of `V_BITS`.
6. Choose the smallest safe value of `C_BITS`.
7. Verify that all data used by the reversible arithmetic are integer-valued.

A typical configuration therefore looks like:

```python
N = 4
L = 2

f = np.array([...])

s_list = [
    np.array([...]),
    np.array([...]),
]

p_pos = np.array([...])
p_neg = np.array([...])

V_BITS = ...
C_BITS = ...
```

---

# Software requirements

The implementation requires:

```text
Python
NumPy
Qiskit
```

For reproducibility, it is recommended to include the exact package versions used for the experiments in a `requirements.txt` file.

For example:

```text
numpy==...
qiskit==...
```

The exact versions used to generate reported results should be recorded before creating a release of the repository.

---

# Reproducibility

When reporting or reproducing an experiment, record at least:

```text
N
L
V_BITS
C_BITS
f
s_list
p_pos
p_neg
threshold
number of Grover iterations
number of adaptive search rounds
random seed
Qiskit version
```

This is sufficient to reconstruct the corresponding problem instance and quantum circuit.
