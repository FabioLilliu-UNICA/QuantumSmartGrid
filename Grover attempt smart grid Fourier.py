import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_qft_full

# ============================================================
# 1. Toy problem
# ============================================================

N = 4       # number of possible shifts
L = 2       # number of shiftable vectors = M - 1

#f = np.array([2, -4, 6, -1])
f = np.array([1, 0, -2, -2])

s_list = [
#    np.array([7, -2, 1, -5]),
#    np.array([-3, 8, -4, 2]),
    np.array([-1, 4, -4, -1]),
    np.array([-3, 3, -1, -2]),
#    np.array([0.5, -0.6, 0.3, -0.2]),
#    np.array([-0.1, 0.3, -0.2, 0.1]),
#    np.array([0.4, -0.7, 0.8, -0.4]),
]

p_pos = np.array([10, 8, 12, 9])
p_neg = np.array([4, 6, 3, 5])


def shifted(s, k):
    return np.roll(s, k)

def total_vector(k_tuple):
    v = f.copy()

    for s, k in zip(s_list, k_tuple):
        v += shifted(s, k)

    return v

def cost(k_tuple):
    v = total_vector(k_tuple)

    v_pos = np.maximum(v, 0.0)
    v_neg = np.minimum(v, 0.0)

    return np.dot(p_pos, v_pos) + np.dot(p_neg, v_neg)

def cost_integer(k_tuple):
    v = total_vector(k_tuple)

    v_pos = np.maximum(v, 0)
    v_neg = np.minimum(v, 0)

    return int(np.dot(p_pos, v_pos) + np.dot(p_neg, v_neg))

def brute_force_solution():
    best_cfg = None
    best_cost = np.inf

    for cfg in itertools.product(range(N), repeat=L):
        c = cost(cfg)

        if c < best_cost:
            best_cost = c
            best_cfg = cfg

    return best_cfg, best_cost

# ============================================================
# 2. Binary encoding of configurations
# ============================================================

SHIFT_BITS = int(np.ceil(np.log2(N)))
K_BITS = L * SHIFT_BITS
NUM_QUBITS = L * SHIFT_BITS
V_BITS = 6                             # enough for v_t in this toy example
C_BITS = 9 #number of bits to encode cost
DIM = 2 ** NUM_QUBITS

k_registers = [
    list(range(i * SHIFT_BITS, (i + 1) * SHIFT_BITS))
    for i in range(L)
]

v_qubits = list(range(K_BITS, K_BITS + V_BITS))
cost_qubits = list(range(K_BITS + V_BITS, K_BITS + V_BITS + C_BITS))

FLAG_QUBIT = K_BITS + V_BITS + C_BITS

    
sign_bit = v_qubits[-1]

assert 2 ** SHIFT_BITS == N, "This simple toy version assumes N is a power of two."


def index_to_config(index):
    """
    Convert a computational-basis index into a shift configuration.

    Example:
        index -> (k_1, k_2, ..., k_L)
    """
    config = []

    for i in range(L):
        k = 0

        for b in range(SHIFT_BITS):
            qubit = i * SHIFT_BITS + b
            bit = (index >> qubit) & 1
            k += bit * (2 ** b)

        config.append(k)

    return tuple(config)


def config_to_index(config):
    """
    Convert a shift configuration into its computational-basis index.
    """
    index = 0

    for i, k in enumerate(config):
        for b in range(SHIFT_BITS):
            bit = (k >> b) & 1
            qubit = i * SHIFT_BITS + b
            index += bit * (2 ** qubit)

    return index

def prepare_basis_state(qc, value, qubits):
    """
    Prepare the integer 'value' on the given qubits, little-endian.
    """
    for b, q in enumerate(qubits):
        if (value >> b) & 1:
            qc.x(q)


def unsigned_to_signed(x, bits):
    """
    Convert unsigned integer x into signed two's-complement value.
    """
    if x >= 2 ** (bits - 1):
        return x - 2 ** bits
    return x


def extract_register_value_from_index(index, qubits):
    """
    Extract the unsigned integer stored in the chosen qubits from
    a computational-basis statevector index.
    """
    value = 0

    for b, q in enumerate(qubits):
        bit = (index >> q) & 1
        value += bit * (2 ** b)

    return value

# ============================================================
# 3. Threshold oracle
# ============================================================

def apply_threshold_oracle(psi, threshold):
    """
    Apply the Grover threshold oracle:

        |k> -> -|k> if C(k) < threshold,
        |k> ->  |k> otherwise.

    This function does NOT precompute all costs.
    It evaluates cost(cfg) on the fly while applying the oracle.

    In a real quantum implementation, this step would be replaced
    by a reversible circuit computing C(k) < threshold.
    """
    psi = psi.copy()

    for index in range(DIM):
        cfg = index_to_config(index)

        if cost(cfg) < threshold:
            psi[index] *= -1.0

    return psi


# ============================================================
# 4. Diffusion operator
# ============================================================

def apply_diffusion(psi):
    """
    Apply the standard Grover diffusion operator:

        D = 2|s><s| - I

    where |s> is the uniform superposition.
    """
    mean_amplitude = np.mean(psi)
    return 2 * mean_amplitude - psi


# ============================================================
# 5. Grover search below a threshold
# ============================================================

def uniform_state():
    return np.ones(DIM, dtype=complex) / np.sqrt(DIM)


def run_grover_for_threshold(threshold, num_iterations):
    """
    Run Grover search for configurations satisfying C(k) < threshold.
    """
    psi = uniform_state()

    for _ in range(num_iterations):
        psi = apply_threshold_oracle(psi, threshold)
        psi = apply_diffusion(psi)

    return psi


def sample_state(psi, rng):
    """
    Sample a computational-basis state from the statevector.
    """
    probs = np.abs(psi) ** 2
    index = rng.choice(DIM, p=probs)
    cfg = index_to_config(index)
    return cfg


# ============================================================
# 6. Grover-style adaptive minimum finding
# ============================================================

def random_config(rng):
    return tuple(rng.integers(0, N, size=L))


def grover_minimize(max_rounds=30, seed=42):
    rng = np.random.default_rng(seed)

    # Initial classical random guess
    best_cfg = random_config(rng)
    best_cost = cost(best_cfg)

    print("Initial configuration:", best_cfg)
    print("Initial cost:", best_cost)

    # BBHT-style adaptive parameter
    m = 1
    growth = 8 / 7
    max_m = int(np.sqrt(DIM))

    for round_id in range(max_rounds):
        # Choose a random number of Grover iterations
        # between 0 and m - 1
        num_iterations = rng.integers(0, m)

        psi = run_grover_for_threshold(
            threshold=best_cost,
            num_iterations=num_iterations
        )

        candidate_cfg = sample_state(psi, rng)
        candidate_cost = cost(candidate_cfg)

        improved = candidate_cost < best_cost

        print(
            f"Round {round_id:02d} | "
            f"Grover iterations: {num_iterations:2d} | "
            f"candidate: {candidate_cfg} | "
            f"cost: {candidate_cost:.4f} | "
            f"improved: {improved}"
        )

        if improved:
            best_cfg = candidate_cfg
            best_cost = candidate_cost
            m = 1
        else:
            m = min(int(np.ceil(growth * m)), max_m)

    return best_cfg, best_cost

# best_cfg_grover, best_cost_grover = grover_minimize(
#     max_rounds=30,
#     seed=42
# )

# print("\nGrover-style result:")
# print("best configuration:", best_cfg_grover)
# print("best cost:", best_cost_grover)


# ============================================================
# 7. Optional brute-force validation
# ============================================================

# true_best_cfg, true_best_cost = brute_force_solution()

# print("\nClassical brute-force optimum:")
# print("best configuration:", true_best_cfg)
# print("best cost:", true_best_cost)


### LOW-LEVEL CIRCUIT PRIMITIVES ###

def append_mcx(qc, controls, target):
    """
    Apply an X gate to target controlled on all controls being 1.
    """
    controls = list(controls)

    if len(controls) == 0:
        qc.x(target)
    elif len(controls) == 1:
        qc.cx(controls[0], target)
    else:
        qc.mcx(controls, target)

def append_qft(qc, qubits):
    """
    Append QFT on the given register, avoiding deprecated QFT class.
    """
    qft_circuit = synth_qft_full(
        num_qubits=len(qubits),
        do_swaps=True,
        approximation_degree=0,
        inverse=False
    )

    qc.append(qft_circuit.to_gate(label="QFT"), qubits)


def append_iqft(qc, qubits):
    """
    Append inverse QFT on the given register, avoiding deprecated QFT class.
    """
    iqft_circuit = synth_qft_full(
        num_qubits=len(qubits),
        do_swaps=True,
        approximation_degree=0,
        inverse=True
    )

    qc.append(iqft_circuit.to_gate(label="IQFT"), qubits)
    
def append_controlled_phase(qc, angle, control_qubits, target_qubit):
    """
    Apply a phase rotation to target_qubit, controlled on all control_qubits.

    If control_qubits is empty, this is just an ordinary phase rotation.
    """

    control_qubits = list(control_qubits)

    # Remove numerically tiny angles.
    angle = float(angle)
    if abs(angle) < 1e-12:
        return

    if len(control_qubits) == 0:
        qc.p(angle, target_qubit)
    elif len(control_qubits) == 1:
        qc.cp(angle, control_qubits[0], target_qubit)
    else:
        qc.mcp(angle, control_qubits, target_qubit)
        
### ADDERS ###

def append_controlled_add_constant_qft(qc, control_qubits, target_qubits, constant):
    """
    Controlled modular addition of a classical constant using QFT.

    If all control_qubits are 1, this applies

        |y> -> |y + constant mod 2^b>

    to target_qubits.

    If control_qubits is empty, this is an ordinary constant addition.

    target_qubits are little-endian.
    """

    control_qubits = list(control_qubits)
    target_qubits = list(target_qubits)

    n = len(target_qubits)
    modulus = 2 ** n

    constant_mod = int(constant) % modulus

    if constant_mod == 0:
        return

    # Move target register to Fourier basis.
    append_qft(qc, target_qubits)

    # In the Fourier basis, adding a constant is diagonal.
    #
    # For a little-endian target register, target_qubits[j]
    # represents the bit 2^j. Therefore the phase angle is:
    #
    #   2*pi*constant*2^j / 2^n.
    #
    for j, q in enumerate(target_qubits):
        angle = 2 * np.pi * constant_mod * (2 ** j) / modulus

        append_controlled_phase(
            qc,
            angle=angle,
            control_qubits=control_qubits,
            target_qubit=q
        )

    # Return to computational basis.
    append_iqft(qc, target_qubits)
    
append_controlled_add_constant = append_controlled_add_constant_qft

def append_add_constant(qc, target_qubits, constant):
    """
    Add a classical constant to target_qubits modulo 2^b.

    This replaces the old dense add_constant_gate.
    """

    append_controlled_add_constant(
        qc,
        control_qubits=[],
        target_qubits=target_qubits,
        constant=constant
    )
    
def append_controlled_add_constant_fourier(qc, control_qubits, target_qubits, constant):
    """
    Controlled addition of a classical constant to a target register
    that is already in Fourier basis.

    This applies only the phase rotations. It does NOT apply QFT/IQFT.

    If all control_qubits are 1, it implements:

        |y>_F -> |y + constant>_F

    where |.>_F denotes Fourier-basis encoding.
    """

    control_qubits = list(control_qubits)
    target_qubits = list(target_qubits)

    n = len(target_qubits)
    modulus = 2 ** n

    constant_mod = int(constant) % modulus

    if constant_mod == 0:
        return

    for j, q in enumerate(target_qubits):
        angle = 2 * np.pi * constant_mod * (2 ** j) / modulus

        append_controlled_phase(
            qc,
            angle=angle,
            control_qubits=control_qubits,
            target_qubit=q
        )
    
### v_t COMPUTATION ###

def selected_constants_for_component(i, t):
    return [int(np.roll(s_list[i], k)[t]) for k in range(N)]


def append_selected_add_constant(qc, shift_qubits, target_qubits, constants):
    """
    Structured reversible implementation of

        |k>|y> -> |k>|y + constants[k] mod 2^b>

    where:
      - shift_qubits encode k in little-endian form,
      - target_qubits store y,
      - constants is a list [c_0, c_1, ..., c_{2^m - 1}].

    This replaces the dense selected_add_constant_gate.
    """

    shift_qubits = list(shift_qubits)
    target_qubits = list(target_qubits)

    num_shifts = 2 ** len(shift_qubits)

    if len(constants) != num_shifts:
        raise ValueError(
            f"Expected {num_shifts} constants, got {len(constants)}."
        )

    for k, constant in enumerate(constants):
        # If constant is 0 mod 2^b, nothing needs to be done.
        modulus = 2 ** len(target_qubits)
        if constant % modulus == 0:
            continue

        # Step 1: convert the condition "shift register == k"
        # into "all shift qubits are 1".
        for b, q in enumerate(shift_qubits):
            bit = (k >> b) & 1
            if bit == 0:
                qc.x(q)

        # Step 2: add the corresponding constant, controlled on all shift bits.
        append_controlled_add_constant(
            qc,
            control_qubits=shift_qubits,
            target_qubits=target_qubits,
            constant=constant
        )

        # Step 3: undo the temporary X gates.
        for b, q in enumerate(shift_qubits):
            bit = (k >> b) & 1
            if bit == 0:
                qc.x(q)
                
def append_selected_add_constant_fourier(
    qc,
    shift_qubits,
    target_qubits,
    constants
):
    """
    Structured reversible implementation of

        |k>|y>_F -> |k>|y + constants[k]>_F

    where the target register is already in Fourier basis.

    No QFT or IQFT is performed inside this function.
    """

    shift_qubits = list(shift_qubits)
    target_qubits = list(target_qubits)

    num_shifts = 2 ** len(shift_qubits)

    if len(constants) != num_shifts:
        raise ValueError(
            f"Expected {num_shifts} constants, got {len(constants)}."
        )

    modulus = 2 ** len(target_qubits)

    for k, constant in enumerate(constants):

        # Nothing to add.
        if constant % modulus == 0:
            continue

        # Convert the condition k_i == k into the standard
        # "all control qubits are 1" condition.
        for b, q in enumerate(shift_qubits):
            bit = (k >> b) & 1
            if bit == 0:
                qc.x(q)

        # The v-register is already in Fourier basis.
        append_controlled_add_constant_fourier(
            qc,
            control_qubits=shift_qubits,
            target_qubits=target_qubits,
            constant=constant
        )

        # Restore the shift register.
        for b, q in enumerate(shift_qubits):
            bit = (k >> b) & 1
            if bit == 0:
                qc.x(q)

def append_compute_v_t_old(qc, t, k_registers, v_qubits):
    """
    Append the reversible computation

        |k_0,k_1,...>|0> -> |k_0,k_1,...>|v_t(k)|

    where

        v_t(k) = f_t + sum_i (r_{k_i}(s_i))_t

    stored modulo 2^V_BITS in v_qubits.
    """

    # Add the fixed component f_t.
    append_add_constant(
        qc,
        target_qubits=v_qubits,
        constant=int(f[t])
    )

    # Add the selected shifted contribution of each s_i.
    for i, k_qubits in enumerate(k_registers):
        constants = selected_constants_for_component(i, t)

        append_selected_add_constant(
            qc,
            shift_qubits=k_qubits,
            target_qubits=v_qubits,
            constants=constants
        )
        
def append_compute_v_t(qc, t, k_registers, v_qubits):
    """
    Append the reversible computation

        |k>|0> -> |k>|v_t(k)>

    using a single QFT/IQFT pair on the v-register.

    The v-register is kept in Fourier basis while all fixed and
    shift-dependent contributions are accumulated.
    """

    # Move v-register to Fourier basis only once.
    append_qft(qc, v_qubits)

    # Add the fixed contribution f(t).
    append_controlled_add_constant_fourier(
        qc,
        control_qubits=[],
        target_qubits=v_qubits,
        constant=int(f[t])
    )

    # Add the selected shifted contribution from every flexible load.
    for i, k_qubits in enumerate(k_registers):

        constants = selected_constants_for_component(i, t)

        append_selected_add_constant_fourier(
            qc,
            shift_qubits=k_qubits,
            target_qubits=v_qubits,
            constants=constants
        )

    # Return v-register to computational basis.
    append_iqft(qc, v_qubits)
        
def compute_v_t_gate(t):
    """
    Build a gate that computes v_t(k) into the v-register.

    Gate input/output layout:
        [all k-register qubits, v-register qubits]

    It implements:
        |k>|0> -> |k>|v_t(k)>.
    """

    num_k_qubits = L * SHIFT_BITS
    total_qubits = num_k_qubits + V_BITS

    sub = QuantumCircuit(total_qubits, name=f"compute_v{t}")

    local_k_registers = [
        list(range(i * SHIFT_BITS, (i + 1) * SHIFT_BITS))
        for i in range(L)
    ]

    local_v_qubits = list(range(num_k_qubits, num_k_qubits + V_BITS))

    append_compute_v_t(
        sub,
        t=t,
        k_registers=local_k_registers,
        v_qubits=local_v_qubits
    )

    return sub.to_gate(label=f"V_{t}")
    
### COST ACCUMULATION ###

def contribution_from_v(t, v_signed):
    """
    Compute the contribution of component t to the cost.

    If v_t >= 0:
        contribution = p_pos[t] * v_t
    else:
        contribution = p_neg[t] * v_t
    """
    if v_signed >= 0:
        return int(p_pos[t] * v_signed)
    else:
        return int(p_neg[t] * v_signed)


def append_add_weighted_bits_fourier(
    qc,
    bit_qubits,
    cost_qubits,
    coefficient,
    extra_controls=None
):
    """
    Add

        coefficient * sum_j 2^j bit_j

    to a cost register that is already in Fourier basis.

    bit_qubits are little-endian.
    """

    if extra_controls is None:
        extra_controls = []

    extra_controls = list(extra_controls)

    for j, bit_qubit in enumerate(bit_qubits):
        weight = coefficient * (2 ** j)

        append_controlled_add_constant_fourier(
            qc,
            control_qubits=extra_controls + [bit_qubit],
            target_qubits=cost_qubits,
            constant=weight
        )
        
def append_add_contribution_t_bitwise_fourier(qc, t, v_qubits, cost_qubits):
    """
    Add contribution_t(v_t) to cost_qubits, assuming cost_qubits
    are already in Fourier basis.

    v_qubits are still computational-basis two's-complement bits.
    """

    sign_bit = v_qubits[-1]
    magnitude_bits = v_qubits[:-1]

    p_plus = int(p_pos[t])
    p_minus = int(p_neg[t])

    # Case 1: v_t >= 0, i.e. sign_bit = 0.
    qc.x(sign_bit)

    append_add_weighted_bits_fourier(
        qc,
        bit_qubits=magnitude_bits,
        cost_qubits=cost_qubits,
        coefficient=p_plus,
        extra_controls=[sign_bit]
    )

    qc.x(sign_bit)

    # Case 2: v_t < 0, i.e. sign_bit = 1.
    append_add_weighted_bits_fourier(
        qc,
        bit_qubits=magnitude_bits,
        cost_qubits=cost_qubits,
        coefficient=p_minus,
        extra_controls=[sign_bit]
    )

    append_controlled_add_constant_fourier(
        qc,
        control_qubits=[sign_bit],
        target_qubits=cost_qubits,
        constant=-p_minus * (2 ** (V_BITS - 1))
    )
    
def append_compute_cost_fourier_accumulation(qc, k_registers, v_qubits, cost_qubits):
    """
    Compute the full cost C(k), but keep cost_qubits in Fourier basis
    during the accumulation.

    Implements:

        |k>|0>_v|0>_C -> |k>|0>_v|C(k)>_C

    with only one QFT and one IQFT on the cost register.
    """

    k_qubits_flat = [q for reg in k_registers for q in reg]

    # Move cost register to Fourier basis once.
    append_qft(qc, cost_qubits)

    for t in range(N):
        v_gate = compute_v_t_gate(t)

        # Compute v_t(k)
        qc.append(v_gate, k_qubits_flat + v_qubits)

        # Add contribution_t(v_t) directly in Fourier basis.
        append_add_contribution_t_bitwise_fourier(
            qc,
            t=t,
            v_qubits=v_qubits,
            cost_qubits=cost_qubits
        )

        # Uncompute v_t(k)
        qc.append(v_gate.inverse(), k_qubits_flat + v_qubits)

    # Return cost register to computational basis.
    append_iqft(qc, cost_qubits)

def compute_cost_gate():
    """
    Build a gate implementing:

        |k>|0>_v|0>_C -> |k>|0>_v|C(k)>_C

    using Fourier-basis accumulation for the cost register.
    """

    total_qubits = K_BITS + V_BITS + C_BITS

    sub = QuantumCircuit(total_qubits, name="compute_cost")

    local_k_registers = [
        list(range(i * SHIFT_BITS, (i + 1) * SHIFT_BITS))
        for i in range(L)
    ]

    local_v_qubits = list(range(K_BITS, K_BITS + V_BITS))
    local_cost_qubits = list(range(K_BITS + V_BITS, K_BITS + V_BITS + C_BITS))

    append_compute_cost_fourier_accumulation(
        sub,
        k_registers=local_k_registers,
        v_qubits=local_v_qubits,
        cost_qubits=local_cost_qubits
    )

    return sub.to_gate(label="Cost")

### COMPARATOR AND PHASE ORACLE ###

def append_compare_less_than_threshold_by_subtraction(
    qc,
    cost_qubits,
    flag_qubit,
    threshold
):
    """
    Flip flag_qubit if unsigned cost register satisfies:

        C < threshold.

    This works by temporarily computing C - threshold,
    copying the sign bit into the flag, and then restoring C.

    Assumption:
        all relevant costs and thresholds lie in the signed-positive range
        of the cost register, i.e.

            0 <= C, threshold < 2^(C_BITS-1).

    cost_qubits are little-endian.
    """

    num_bits = len(cost_qubits)
    positive_limit = 2 ** (num_bits - 1)

#    if threshold < 0:
#        raise ValueError("Threshold should be nonnegative.")

    if threshold >= positive_limit:
        raise ValueError(
            f"Threshold {threshold} is too large for sign-based comparison "
            f"with {num_bits} bits. Need threshold < {positive_limit}."
        )

    # Temporarily compute C - threshold.
    append_add_constant(
        qc,
        target_qubits=cost_qubits,
        constant=-threshold
    )

    # If C - threshold is negative, the sign bit is 1.
    sign_bit = cost_qubits[-1]
    qc.cx(sign_bit, flag_qubit)

    # Restore C.
    append_add_constant(
        qc,
        target_qubits=cost_qubits,
        constant=threshold
    )
    
def append_threshold_phase_oracle(qc, threshold):
    """
    Append the Grover threshold phase oracle:

        |k>|0>_v|0>_C|0>_F
        ->
        (-1)^[C(k) < threshold] |k>|0>_v|0>_C|0>_F.

    The flag, cost register, and v-register are all uncomputed at the end.
    """

    flag_qubit = K_BITS + V_BITS + C_BITS

    all_work_qubits = (
        [q for reg in k_registers for q in reg]
        + v_qubits
        + cost_qubits
    )

    cost_gate = compute_cost_gate()

    # 1. Compute C(k)
    qc.append(cost_gate, all_work_qubits)

    # 2. Compute flag = [C(k) < threshold]
    append_compare_less_than_threshold_by_subtraction(
        qc,
        cost_qubits=cost_qubits,
        flag_qubit=flag_qubit,
        threshold=threshold
    )

    # 3. Apply phase flip if flag = 1
    qc.z(flag_qubit)

    # 4. Uncompute flag
    append_compare_less_than_threshold_by_subtraction(
        qc,
        cost_qubits=cost_qubits,
        flag_qubit=flag_qubit,
        threshold=threshold
    )

    # 5. Uncompute C(k)
    qc.append(cost_gate.inverse(), all_work_qubits)
    
### DIFFUSION AND GROVER CIRCUIT ###

def append_diffusion_on_k(qc, k_qubits):
    """
    Append the Grover diffusion operator on the configuration register.

    It implements, up to global phase:

        D = 2|s><s| - I

    on the k-register only.
    """

    # H and X on all k-qubits
    for q in k_qubits:
        qc.h(q)
        qc.x(q)

    # Multi-controlled Z on |11...1>
    target = k_qubits[-1]
    controls = k_qubits[:-1]

    qc.h(target)
    append_mcx(qc, controls, target)
    qc.h(target)

    # Undo X and H
    for q in reversed(k_qubits):
        qc.x(q)
        qc.h(q)
        
k_qubits_flat = [q for reg in k_registers for q in reg]

def append_grover_iteration(qc, threshold):
    """
    Append one Grover iteration:

        G = D O_T.
    """

    k_qubits_flat = [q for reg in k_registers for q in reg]

    append_threshold_phase_oracle(qc, threshold)

    append_diffusion_on_k(qc, k_qubits_flat)
    
def marked_probability_from_state(state, threshold):
    cfg_probs = configuration_probabilities_from_state(state)

    total_prob = 0.0

    for cfg, prob in cfg_probs.items():
        if cost_integer(cfg) < threshold:
            total_prob += prob

    return total_prob

def run_grover_circuit(threshold, num_iterations):
    """
    Build and simulate Grover search using the real threshold phase oracle.
    """

    num_qubits = K_BITS + V_BITS + C_BITS + 1
    qc = QuantumCircuit(num_qubits)

    k_qubits_flat = [q for reg in k_registers for q in reg]

    # Uniform superposition over configurations
    for q in k_qubits_flat:
        qc.h(q)

    # Grover iterations
    for _ in range(num_iterations):
        append_grover_iteration(qc, threshold)

    state = Statevector.from_instruction(qc)

    return state


### GROVER WITH REAL ORACLE ###

def configuration_probabilities_from_state(state):
    """
    Return probabilities on configurations, summing only clean work-register states.
    """

    probs = state.probabilities()
    flag_qubit = K_BITS + V_BITS + C_BITS

    cfg_probs = {}

    for index, prob in enumerate(probs):
        v_val = extract_register_value_from_index(index, v_qubits)
        c_val = extract_register_value_from_index(index, cost_qubits)
        flag = (index >> flag_qubit) & 1

        if v_val != 0 or c_val != 0 or flag != 0:
            continue

        k_tuple = tuple(
            extract_register_value_from_index(index, reg)
            for reg in k_registers
        )

        cfg_probs[k_tuple] = cfg_probs.get(k_tuple, 0.0) + prob

    return cfg_probs

def sample_configuration_from_state(state, rng):
    """
    Sample a configuration from the clean work-register part of the state.
    """

    cfg_probs = configuration_probabilities_from_state(state)

    configs = list(cfg_probs.keys())
    probs = np.array([cfg_probs[cfg] for cfg in configs], dtype=float)

    # Numerical safety: normalize clean-register probabilities.
    probs = probs / np.sum(probs)

    choice = rng.choice(len(configs), p=probs)

    return configs[choice]

def grover_minimize_with_real_oracle(max_rounds=8, seed=42):
    """
    Adaptive Grover-style minimum finding using the reversible threshold oracle.

    Warning:
        This is computationally expensive in statevector simulation.
        Use small max_rounds for now.
    """

    rng = np.random.default_rng(seed)

    # Initial classical random guess
    best_cfg = random_config(rng)
    best_cost = cost_integer(best_cfg)

    print("Initial configuration:", best_cfg)
    print("Initial cost:", best_cost)

    m = 1
    growth = 8 / 7
    max_m = int(np.ceil(np.sqrt(DIM)))

    for round_id in range(max_rounds):
        num_iterations = int(rng.integers(0, m))

        state = run_grover_circuit(
            threshold=best_cost,
            num_iterations=num_iterations
        )

        candidate_cfg = sample_configuration_from_state(state, rng)
        candidate_cost = cost_integer(candidate_cfg)

        improved = candidate_cost < best_cost

        print(
            f"Round {round_id:02d} | "
            f"threshold: {best_cost:3d} | "
            f"iterations: {num_iterations:2d} | "
            f"candidate: {candidate_cfg} | "
            f"cost: {candidate_cost:3d} | "
            f"improved: {improved}"
        )

        if improved:
            best_cfg = candidate_cfg
            best_cost = candidate_cost
            m = 1
        else:
            m = min(int(np.ceil(growth * m)), max_m)

    return best_cfg, best_cost
    
### TESTS ###

def circuit_summary(qc, name="circuit"):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("num qubits:", qc.num_qubits)
    print("depth:", qc.depth())
    print("size:", qc.size())
    print("ops:")
    for gate, count in qc.count_ops().items():
        print(f"  {gate}: {count}")
        
def build_cost_test_circuit(k_tuple=None):
    num_qubits = K_BITS + V_BITS + C_BITS
    qc = QuantumCircuit(num_qubits)

    if k_tuple is not None:
        for i, k_i in enumerate(k_tuple):
            prepare_basis_state(qc, k_i, k_registers[i])

    append_compute_cost_fourier_accumulation(
        qc,
        k_registers=k_registers,
        v_qubits=v_qubits,
        cost_qubits=cost_qubits
    )

    return qc

def build_phase_oracle_test_circuit(threshold):
    num_qubits = K_BITS + V_BITS + C_BITS + 1
    qc = QuantumCircuit(num_qubits)

    append_threshold_phase_oracle(qc, threshold)

    return qc

def build_one_grover_iteration_circuit(threshold):
    num_qubits = K_BITS + V_BITS + C_BITS + 1
    qc = QuantumCircuit(num_qubits)

    append_grover_iteration(qc, threshold)

    return qc

import time

def benchmark_grover_runtime(threshold, max_iterations=3):
    rows = []

    for r in range(max_iterations + 1):
        start = time.perf_counter()

        state = run_grover_circuit(
            threshold=threshold,
            num_iterations=r
        )

        elapsed = time.perf_counter() - start
        p_marked = marked_probability_from_state(state, threshold)

        rows.append((r, elapsed, p_marked))

        print(
            f"iterations={r}, "
            f"time={elapsed:.3f}s, "
            f"marked probability={p_marked:.6f}"
        )

    return rows

def theoretical_grover_probability(num_marked, search_space_size, r):
    theta = np.arcsin(np.sqrt(num_marked / search_space_size))
    return np.sin((2 * r + 1) * theta) ** 2

def circuit_summary_decomposed(qc, name="circuit", reps=1):
    qc_dec = qc.decompose(reps=reps)

    print("\n" + "=" * 60)
    print(f"{name}, decomposed reps={reps}")
    print("=" * 60)
    print("num qubits:", qc_dec.num_qubits)
    print("depth:", qc_dec.depth())
    print("size:", qc_dec.size())
    print("ops:")
    for gate, count in qc_dec.count_ops().items():
        print(f"  {gate}: {count}")

    return qc_dec


# qc_oracle = build_phase_oracle_test_circuit(threshold=90)
# qc_oracle_dec = circuit_summary_decomposed(
#     qc_oracle,
#     "Threshold phase oracle",
#     reps=2
# )

### EXECUTION ###
from itertools import product

def deterministic_minimize():
    best_cfg = None
    best_cost = None

    for cfg in product(range(N), repeat=L):
        cost = cost_integer(cfg)

        if best_cost is None or cost < best_cost:
            best_cfg = cfg
            best_cost = cost

    return best_cfg, best_cost


if __name__ == "__main__":

    print("\n=== Deterministic brute-force optimum ===")
    det_cfg, det_cost = deterministic_minimize()
    print("Best configuration:", det_cfg)
    print("Best cost:", det_cost)

    print("\n=== Grover adaptive search ===")
    grover_cfg, grover_cost = grover_minimize_with_real_oracle(
        max_rounds=8,
        seed=42
    )
    print("Best configuration found:", grover_cfg)
    print("Best cost found:", grover_cost)