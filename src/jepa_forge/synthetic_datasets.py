"""Ten distinct, deterministic synthetic families for a broader pilot.

These are controlled software benchmarks, not validated scientific simulators.
All labels and entity identities remain outside the feature matrix. Parameters,
equations, and adaptation boundaries are documented in
``docs/SYNTHETIC_DATASETS_EXPANDED.md`` and each dataset's metadata.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .schema import RawDataset, TaskSpec


SYNTHETIC_DATASET_NAMES = (
    "synthetic_lorenz",
    "synthetic_mackey_glass",
    "synthetic_switching_var",
    "synthetic_chirp_seasonal",
    "synthetic_coupled_oscillators",
    "synthetic_nonlinear_ar",
    "synthetic_nonlinear_multiview",
    "synthetic_hierarchical_multiclass",
    "synthetic_sparse_interactions",
    "synthetic_manifold_nuisance",
)

_N_ENTITIES = 100
_WINDOWS_PER_ENTITY = 20
_WINDOW = 32
_STRIDE = 8
_LENGTH = _WINDOW + (_WINDOWS_PER_ENTITY - 1) * _STRIDE
_N_ROWS = _N_ENTITIES * _WINDOWS_PER_ENTITY
_LORENZ_URL = "https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml"
_MACKEY_GLASS_URL = "https://doi.org/10.1126/science.267326"


def _base_metadata(name: str, seed: int, mechanism: str, parameters: dict,
                   source_url: str | None = None) -> dict:
    return {
        "source": "Locally generated synthetic benchmark; no external observations",
        "source_url": source_url,
        "license": "CC0-1.0",
        "generator_seed": int(seed),
        "generator_family": name.removeprefix("synthetic_"),
        "generator_version": 1,
        "generator": mechanism,
        "generator_parameters": parameters,
        "scientific_limitation": "Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.",
    }


def _series_dataset(name: str, seed: int, sequences: np.ndarray, mechanism: str,
                    parameters: dict, source_url: str | None = None) -> RawDataset:
    n_entities, length, channels = sequences.shape
    if n_entities != _N_ENTITIES or length != _LENGTH or not np.isfinite(sequences).all():
        raise ValueError("Generator returned invalid entity trajectories")
    rows, groups, intervals = [], [], []
    for entity in range(_N_ENTITIES):
        for window in range(_WINDOWS_PER_ENTITY):
            start = window * _STRIDE
            rows.append(sequences[entity, start:start + _WINDOW].reshape(-1))
            groups.append(entity)
            intervals.append([start, start + _WINDOW - 1])
    names = [f"t{t:02d}_channel{channel}" for t in range(_WINDOW) for channel in range(channels)]
    metadata = _base_metadata(name, seed, mechanism, {
        "entities": _N_ENTITIES, "windows_per_entity": _WINDOWS_PER_ENTITY,
        "window_timesteps": _WINDOW, "stride": _STRIDE, "channels": channels,
        "retained_timesteps_per_entity": _LENGTH, **parameters,
    }, source_url)
    metadata.update({
        "input_shape": [_WINDOW, channels], "flatten_order": "time-major",
        "feature_time_indices": np.repeat(np.arange(_WINDOW), channels).tolist(),
        "task_type": "forecasting", "temporal_direction": "past_to_future",
        "interval_semantics": "inclusive retained raw timestep coverage within entity",
        "split_policy": "held-out independent entities; every overlapping window from one entity stays in one split",
        "group_role": "split_metadata_only", "label_role": "absent",
    })
    return RawDataset(name, "time_series", np.asarray(rows, dtype=np.float64), names,
                      groups=np.asarray(groups, dtype=np.int64),
                      intervals=np.asarray(intervals, dtype=np.int64),
                      roles={feature: "measurement" for feature in names}, metadata=metadata)


def _tabular_dataset(name: str, seed: int, view_a: np.ndarray, view_b: np.ndarray,
                     labels: np.ndarray, mechanism: str, parameters: dict) -> RawDataset:
    X = np.concatenate((view_a, view_b), axis=1).astype(np.float64)
    if X.shape[0] != _N_ROWS or not np.isfinite(X).all():
        raise ValueError("Generator returned invalid independent observations")
    names = [f"view_a_{i:02d}" for i in range(view_a.shape[1])]
    names += [f"view_b_{i:02d}" for i in range(view_b.shape[1])]
    metadata = _base_metadata(name, seed, mechanism, {"rows": _N_ROWS, **parameters})
    metadata.update({
        "input_shape": [X.shape[1]], "task_type": "classification",
        "label_role": "evaluation_only", "label_names": [f"class_{c}" for c in np.unique(labels)],
        "label_storage": "Separate RawDataset.y only; labels are absent from X and feature roles",
        "view_a_columns": list(range(view_a.shape[1])),
        "view_b_columns": list(range(view_a.shape[1], X.shape[1])),
        "split_policy": "random independent rows, independent of labels",
    })
    return RawDataset(name, "tabular", X, names, y=np.asarray(labels, dtype=np.int64),
                      roles={feature: "measurement" for feature in names}, metadata=metadata)


def _lorenz(rng: np.random.Generator, seed: int) -> RawDataset:
    rho = rng.uniform(26.0, 30.0, _N_ENTITIES)
    state = rng.uniform([-12.0, -12.0, 8.0], [12.0, 12.0, 28.0], (_N_ENTITIES, 3))
    dt, burn = 0.02, 500

    def derivative(q):
        x, y, z = q.T
        return np.column_stack((10.0 * (y - x), x * (rho - z) - y, x * y - (8.0 / 3.0) * z))

    observations = []
    for step in range(burn + _LENGTH):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if step >= burn:
            observations.append(state.copy())
    sequence = np.stack(observations, axis=1)
    sequence += rng.normal(0, 0.05, sequence.shape)
    return _series_dataset("synthetic_lorenz", seed, sequence,
        "Lorenz differential equations integrated by RK4; independent initial states/rho and noisy three-state observations",
        {"equations": ["dx/dt=10(y-x)", "dy/dt=x(rho-z)-y", "dz/dt=x*y-(8/3)z"],
         "rho_range": [26.0, 30.0], "sigma": 10.0, "beta": 8.0 / 3.0,
         "dt": dt, "burn_in_steps": burn, "measurement_noise_std": 0.05,
         "initial_state_min": [-12.0, -12.0, 8.0], "initial_state_max": [12.0, 12.0, 28.0],
         "adaptation": "Entity rho variation, observation noise, RK4, and windowing are pilot choices, not a reproduction of the original paper."}, _LORENZ_URL)


def _mackey_glass(rng: np.random.Generator, seed: int) -> RawDataset:
    delay = rng.integers(14, 23, _N_ENTITIES)
    beta, gamma = rng.uniform(0.18, 0.23, _N_ENTITIES), rng.uniform(0.08, 0.12, _N_ENTITIES)
    history, burn = 23, 400
    total = history + burn + 8 + _LENGTH
    values = np.zeros((_N_ENTITIES, total))
    values[:, :history] = rng.uniform(0.8, 1.3, (_N_ENTITIES, 1)) + rng.normal(0, 0.02, (_N_ENTITIES, history))
    entities = np.arange(_N_ENTITIES)
    for t in range(history, total):
        delayed = values[entities, t - 1 - delay]
        innovation = rng.normal(0, 0.002, _N_ENTITIES)
        values[:, t] = np.maximum(0.01, (1 - gamma) * values[:, t - 1] + beta * delayed / (1 + delayed ** 10) + innovation)
    start = history + burn + 8
    current = values[:, start:start + _LENGTH]
    lag4, lag8 = values[:, start - 4:start - 4 + _LENGTH], values[:, start - 8:start - 8 + _LENGTH]
    sequence = np.stack((current, lag4, current ** 2 + 0.2 * lag8), axis=2)
    sequence += rng.normal(0, 0.005, sequence.shape)
    return _series_dataset("synthetic_mackey_glass", seed, sequence,
        "Noisy discrete Euler Mackey-Glass-inspired delayed feedback with three causal observation channels",
        {"equation": "x[t]=max(.01,(1-gamma)*x[t-1]+beta*x[t-1-delay]/(1+x[t-1-delay]^10)+epsilon)",
         "dt": 1.0, "delay_integer_range_inclusive": [14, 22], "beta_range": [0.18, 0.23],
         "gamma_range": [0.08, 0.12], "exponent": 10, "burn_in_steps": burn,
         "initial_history_range": [0.8, 1.3], "initial_history_noise_std": 0.02,
         "innovation_std": 0.002, "measurement_noise_std": 0.005,
         "observations": ["x[t]", "x[t-4]", "x[t]^2+.2*x[t-8]"],
         "adaptation": "Discrete dt=1 recursion, parameter ranges, positivity floor, process noise and sensor transforms are deliberate adaptations; not a validated DDE solver."}, _MACKEY_GLASS_URL)


def _switching_var(rng: np.random.Generator, seed: int) -> RawDataset:
    matrices = np.asarray([
        [[0.70, 0.15, 0, 0], [-0.10, 0.65, 0.10, 0], [0, -0.15, 0.60, 0.15], [0.10, 0, 0, 0.65]],
        [[0.40, -0.35, 0, 0.10], [0.30, 0.35, 0.10, 0], [0.10, 0, 0.50, -0.25], [0, 0.10, 0.25, 0.45]],
        [[-0.45, 0.15, 0, 0], [0.10, -0.40, 0.20, 0], [0, 0.10, -0.35, 0.20], [0.15, 0, 0.10, -0.40]],
    ])
    offsets = np.asarray([[0.10, -0.08, 0.05, 0.0], [-0.12, 0.08, 0, 0.10], [0.0, 0, -0.10, -0.08]])
    state = rng.normal(0, 0.3, (_N_ENTITIES, 4))
    regime = rng.integers(0, 3, _N_ENTITIES)
    propensity = rng.uniform(0.025, 0.09, _N_ENTITIES)
    entity_scale = rng.uniform(0.8, 1.3, (_N_ENTITIES, 1))
    burn, observations = 100, []
    for step in range(burn + _LENGTH):
        switch = rng.random(_N_ENTITIES) < propensity
        regime[switch] = (regime[switch] + rng.integers(1, 3, switch.sum())) % 3
        state = np.einsum("nij,nj->ni", matrices[regime], state) + offsets[regime]
        state += rng.normal(0, 0.11, state.shape) * entity_scale
        if step >= burn:
            observations.append(state.copy())
    sequence = np.stack(observations, axis=1)
    sequence += rng.normal(0, 0.025, sequence.shape)
    return _series_dataset("synthetic_switching_var", seed, sequence,
        "Three-regime four-channel vector autoregression with persistent latent Markov switching and entity-specific noise scale",
        {"equation": "x[t]=A[regime[t]]@x[t-1]+b[regime[t]]+entity_scale*epsilon[t]",
         "transition_rule": "With entity probability p switch uniformly to one of the two other regimes; otherwise retain regime",
         "switch_probability_range": [0.025, 0.09], "transition_matrices": matrices.tolist(),
         "regime_offsets": offsets.tolist(), "entity_noise_scale_range": [0.8, 1.3],
         "innovation_std": 0.11, "measurement_noise_std": 0.025, "burn_in_steps": burn,
         "latent_regime_is_input": False})


def _chirp_seasonal(rng: np.random.Generator, seed: int) -> RawDataset:
    t = np.arange(_LENGTH)[None, :]
    frequency = rng.uniform(0.012, 0.035, (_N_ENTITIES, 1))
    drift = rng.uniform(0.00004, 0.00018, (_N_ENTITIES, 1))
    phase = rng.uniform(-np.pi, np.pi, (_N_ENTITIES, 1))
    amplitude = rng.uniform(0.7, 1.5, (_N_ENTITIES, 1))
    trend = rng.uniform(-0.004, 0.004, (_N_ENTITIES, 1)) * (t - _LENGTH / 2)
    angle = 2 * np.pi * (frequency * t + 0.5 * drift * t ** 2) + phase
    envelope = amplitude * (1 + 0.25 * np.sin(2 * np.pi * t / 90 + phase))
    season = 0.35 * np.sin(2 * np.pi * t / 18 + phase / 2)
    innovations = rng.normal(0, 0.04, (_N_ENTITIES, _LENGTH))
    colored = np.zeros_like(innovations)
    for step in range(1, _LENGTH):
        colored[:, step] = 0.75 * colored[:, step - 1] + innovations[:, step]
    sequence = np.stack((envelope * np.sin(angle) + season + trend + colored,
                         0.8 * envelope * np.cos(angle + 0.3) - 0.5 * season + 0.7 * trend,
                         0.5 * envelope * np.sin(2 * angle) + 0.8 * season - 0.4 * trend + 0.5 * colored), axis=2)
    sequence += rng.normal(0, 0.03, sequence.shape)
    return _series_dataset("synthetic_chirp_seasonal", seed, sequence,
        "Nonstationary chirp phase plus slow amplitude modulation, fixed seasonal component, linear trend and colored noise",
        {"chirp_phase": "2*pi*(f0*t+.5*drift*t^2)+phase", "initial_frequency_range": [0.012, 0.035],
         "frequency_drift_range": [0.00004, 0.00018], "amplitude_range": [0.7, 1.5],
         "envelope": "amplitude*(1+.25*sin(2*pi*t/90+phase))", "season": ".35*sin(2*pi*t/18+phase/2)",
         "slope_range": [-0.004, 0.004], "ar_coefficient": 0.75, "ar_innovation_std": 0.04,
         "measurement_noise_std": 0.03, "phase_range": [-float(np.pi), float(np.pi)]})


def _coupled_oscillators(rng: np.random.Generator, seed: int) -> RawDataset:
    dt, burn = 0.08, 250
    position = rng.normal(0, 0.7, (_N_ENTITIES, 2))
    velocity = rng.normal(0, 0.2, (_N_ENTITIES, 2))
    frequency = rng.uniform(0.7, 1.2, (_N_ENTITIES, 2))
    coupling = rng.uniform(0.12, 0.35, (_N_ENTITIES, 1))
    drive_frequency = rng.uniform(0.5, 0.9, (_N_ENTITIES, 1))
    phase = rng.uniform(-np.pi, np.pi, (_N_ENTITIES, 2))
    observations = []
    for step in range(burn + _LENGTH):
        drive = 0.35 * np.sin(drive_frequency * step * dt + phase)
        acceleration = -frequency ** 2 * position - 0.12 * position ** 3 - 0.10 * velocity
        acceleration += coupling * (position[:, ::-1] - position) + drive
        velocity += dt * acceleration + rng.normal(0, 0.008, velocity.shape)
        position += dt * velocity
        if step >= burn:
            observations.append(np.column_stack((position, velocity)))
    sequence = np.stack(observations, axis=1)
    sequence += rng.normal(0, 0.015, sequence.shape)
    return _series_dataset("synthetic_coupled_oscillators", seed, sequence,
        "Two bidirectionally coupled damped cubic oscillators with independent driving phases and noisy semi-implicit Euler integration",
        {"acceleration": "-omega_i^2*q_i-.12*q_i^3-.10*v_i+k*(q_j-q_i)+.35*sin(omega_drive*t+phase_i)",
         "integration": "v_next=v+dt*acceleration+Normal(0,.008); q_next=q+dt*v_next",
         "dt": dt, "natural_frequency_range": [0.7, 1.2], "coupling_range": [0.12, 0.35],
         "drive_frequency_range": [0.5, 0.9], "channels": 4, "channel_order": ["q0", "q1", "v0", "v1"],
         "measurement_noise_std": 0.015, "burn_in_steps": burn,
         "adaptation": "Authored coupled cubic oscillator benchmark; numerical trajectories are not physical validation."})


def _nonlinear_ar(rng: np.random.Generator, seed: int) -> RawDataset:
    burn = 200
    values = rng.normal(0, 0.3, (_N_ENTITIES, burn + _LENGTH + 3, 3))
    entity_bias = rng.normal(0, 0.10, (_N_ENTITIES, 3))
    gain = rng.uniform(0.75, 1.15, (_N_ENTITIES, 1))
    for step in range(3, values.shape[1]):
        p1, p2, p3 = values[:, step - 1], values[:, step - 2], values[:, step - 3]
        next_values = np.column_stack((
            0.60 * p1[:, 0] + 0.40 * np.tanh(p2[:, 1] * p1[:, 2]),
            0.45 * p1[:, 1] + 0.35 * np.sin(p1[:, 0]) - 0.15 * p3[:, 2],
            0.50 * p1[:, 2] + 0.30 * np.tanh(p1[:, 0] - p2[:, 1]),
        ))
        sigma = 0.06 + 0.06 * np.abs(np.tanh(p1))
        values[:, step] = gain * next_values + entity_bias + rng.normal(size=next_values.shape) * sigma
    sequence = values[:, -_LENGTH:]
    sequence += rng.normal(0, 0.01, sequence.shape)
    return _series_dataset("synthetic_nonlinear_ar", seed, sequence,
        "Three-channel nonlinear autoregression with multiplicative cross-lag interactions and state-dependent innovation variance",
        {"equations_before_gain_bias_noise": ["x[t]=.60*x[t-1]+.40*tanh(y[t-2]*z[t-1])",
          "y[t]=.45*y[t-1]+.35*sin(x[t-1])-.15*z[t-3]", "z[t]=.50*z[t-1]+.30*tanh(x[t-1]-y[t-2])"],
         "entity_gain_range": [0.75, 1.15], "entity_bias_std": 0.10,
         "innovation_std": ".06+.06*abs(tanh(previous_channel_state))",
         "measurement_noise_std": 0.01, "burn_in_steps": burn, "maximum_lag": 3})


def _projection(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    weights = rng.normal(size=(inputs, outputs))
    return weights / np.sqrt(np.square(weights).sum(axis=0, keepdims=True))


def _nonlinear_multiview(rng: np.random.Generator, seed: int) -> RawDataset:
    z = rng.normal(size=(_N_ROWS, 6))
    a = np.tanh(z @ _projection(rng, 6, 32))
    a += 0.15 * (z[:, 0] * z[:, 1])[:, None] * rng.normal(size=(1, 32))
    b = np.sin(z @ _projection(rng, 6, 32))
    b += 0.20 * (z[:, 2] ** 2 - z[:, 3] ** 2)[:, None] * rng.normal(size=(1, 32))
    a += rng.normal(0, 0.08, a.shape)
    b += rng.normal(0, 0.08, b.shape)
    scores = np.column_stack((z[:, 0] + 0.8 * z[:, 1] * z[:, 2],
                              -z[:, 0] + 0.6 * np.sin(z[:, 3]),
                              z[:, 4] - 0.5 * z[:, 5] + 0.4 * z[:, 1] ** 2))
    return _tabular_dataset("synthetic_nonlinear_multiview", seed, a, b, scores.argmax(axis=1),
        "Two nonlinear noisy measurements of six shared Gaussian latent factors; three evaluation classes from latent nonlinear scores",
        {"latent_dimensions": 6, "view_dimensions": [32, 32], "measurement_noise_std": 0.08,
         "view_a": "tanh(z@unit_column_A)+.15*z0*z1*w_a+noise",
         "view_b": "sin(z@unit_column_B)+.20*(z2^2-z3^2)*w_b+noise",
         "class_rule": "argmax(z0+.8*z1*z2, -z0+.6*sin(z3), z4-.5*z5+.4*z1^2)"})


def _hierarchical_multiclass(rng: np.random.Generator, seed: int) -> RawDataset:
    labels = rng.integers(0, 6, _N_ROWS)
    angles = 2 * np.pi * np.arange(3) / 3
    centers = np.zeros((3, 8))
    centers[:, :2] = 2.2 * np.column_stack((np.cos(angles), np.sin(angles)))
    child_directions = _projection(rng, 8, 3).T
    z = centers[labels // 2] + (2 * (labels % 2) - 1)[:, None] * 0.9 * child_directions[labels // 2]
    z += rng.normal(0, 0.45, z.shape)
    a = z @ _projection(rng, 8, 40)
    b = np.tanh(z @ _projection(rng, 8, 40))
    b += 0.12 * (z[:, 2] * z[:, 3])[:, None] * rng.normal(size=(1, 40))
    a += rng.normal(0, 0.12, a.shape)
    b += rng.normal(0, 0.12, b.shape)
    return _tabular_dataset("synthetic_hierarchical_multiclass", seed, a, b, labels,
        "Six latent Gaussian subclasses nested under three separated parent centers, observed through linear and nonlinear views",
        {"classes": 6, "parent_classes": 3, "subclasses_per_parent": 2,
         "latent_dimensions": 8, "parent_circle_radius": 2.2, "child_offset": 0.9,
         "latent_noise_std": 0.45, "measurement_noise_std": 0.12, "view_dimensions": [40, 40],
         "latent_rule": "z=parent_center[y//2]+(2*(y%2)-1)*.9*parent_child_direction+Normal(0,.45)",
         "view_a": "z@unit_column_A+noise", "view_b": "tanh(z@unit_column_B)+.12*z2*z3*w+noise",
         "label_note": "Class IDs determine latent mixture membership only; no class ID or parent ID is appended to X."})


def _sparse_interactions(rng: np.random.Generator, seed: int) -> RawDataset:
    z = rng.normal(size=(_N_ROWS, 10))
    pairs = [(i, (i + 1) % 10) for i in range(10)] + [(i, i + 4) for i in range(6)]
    interactions = np.column_stack([z[:, i] * z[:, j] for i, j in pairs])
    a = np.column_stack((z + rng.normal(0, 0.04, z.shape), rng.normal(size=(_N_ROWS, 38))))
    b = np.column_stack((interactions + rng.normal(0, 0.06, interactions.shape), rng.normal(size=(_N_ROWS, 32))))
    score = z[:, 0] * z[:, 1] + 0.8 * z[:, 2] * z[:, 3] - 0.5 * z[:, 4] + 0.35 * np.sin(z[:, 5])
    labels = (score + rng.normal(0, 0.1, _N_ROWS) > 0).astype(np.int64)
    return _tabular_dataset("synthetic_sparse_interactions", seed, a, b, labels,
        "Sparse nonlinear classification signal among 70 independent nuisance columns; source factors and pairwise products occupy separate views",
        {"latent_dimensions": 10, "view_dimensions": [48, 48], "view_a_signal_columns": 10,
         "view_b_interaction_columns": 16, "independent_nuisance_columns": 70,
         "interaction_pairs": [list(pair) for pair in pairs], "source_noise_std": 0.04,
         "interaction_noise_std": 0.06, "label_noise_std": 0.1,
         "class_rule": "1[z0*z1+.8*z2*z3-.5*z4+.35*sin(z5)+Normal(0,.1)>0]"})


def _manifold_nuisance(rng: np.random.Generator, seed: int) -> RawDataset:
    theta = rng.uniform(1.5 * np.pi, 4.5 * np.pi, _N_ROWS)
    height = rng.uniform(-1.0, 1.0, _N_ROWS)
    intrinsic = np.column_stack((theta * np.cos(theta) / 8, height, theta * np.sin(theta) / 8))
    signal_a = intrinsic @ _projection(rng, 3, 16)
    warped = np.column_stack((np.sin(theta), np.cos(theta), height, height * np.sin(theta)))
    signal_b = np.tanh(warped @ _projection(rng, 4, 16))
    nuisance_a = 2.5 * rng.normal(size=(_N_ROWS, 4)) @ _projection(rng, 4, 32)
    nuisance_b = 2.5 * rng.normal(size=(_N_ROWS, 4)) @ _projection(rng, 4, 32)
    a = np.column_stack((signal_a + rng.normal(0, 0.08, signal_a.shape),
                         nuisance_a + rng.normal(0, 0.2, nuisance_a.shape)))
    b = np.column_stack((signal_b + rng.normal(0, 0.08, signal_b.shape),
                         nuisance_b + rng.normal(0, 0.2, nuisance_b.shape)))
    sector = np.minimum(2, ((theta - 1.5 * np.pi) / np.pi).astype(int))
    labels = 2 * sector + (height > 0).astype(int)
    return _tabular_dataset("synthetic_manifold_nuisance", seed, a, b, labels,
        "Rolled two-dimensional manifold observed through two different embeddings with 64 independent-of-label correlated nuisance measurements",
        {"intrinsic_dimensions": 2, "view_dimensions": [48, 48], "signal_columns_per_view": 16,
         "nuisance_columns_per_view": 32, "theta_range": [1.5 * float(np.pi), 4.5 * float(np.pi)],
         "height_range": [-1.0, 1.0], "view_a_manifold": "(theta*cos(theta)/8, height, theta*sin(theta)/8)",
         "view_b_basis": "(sin(theta),cos(theta),height,height*sin(theta))",
         "nuisance_latent_dimensions_per_view": 4, "nuisance_scale": 2.5,
         "signal_noise_std": 0.08, "nuisance_noise_std": 0.2,
         "class_rule": "2*floor((theta-1.5*pi)/pi)+1[height>0]",
         "nuisance_note": "Each view has independently drawn nuisance factors; abundance remains a distractor after train-fitted per-column scaling."})


_GENERATORS: dict[str, Callable[[np.random.Generator, int], RawDataset]] = dict(zip(
    SYNTHETIC_DATASET_NAMES,
    (_lorenz, _mackey_glass, _switching_var, _chirp_seasonal, _coupled_oscillators,
     _nonlinear_ar, _nonlinear_multiview, _hierarchical_multiclass, _sparse_interactions,
     _manifold_nuisance), strict=True))


def load_synthetic_dataset(name: str, seed: int = 2026) -> RawDataset:
    """Generate exactly one named mechanism with a local RNG and no network I/O."""
    try:
        generator = _GENERATORS[name]
    except KeyError as error:
        raise ValueError(f"Unknown expanded synthetic dataset {name!r}") from error
    return generator(np.random.default_rng(seed), int(seed))


def propose_synthetic_tasks(dataset: RawDataset) -> list[TaskSpec]:
    """Two predeclared masks per family; never inspect labels or held-out values."""
    if dataset.name not in SYNTHETIC_DATASET_NAMES:
        raise ValueError(f"No expanded synthetic task templates for {dataset.name!r}")
    if dataset.modality == "time_series":
        timesteps, channels = dataset.metadata["input_shape"]
        return [TaskSpec(f"past{past}_future{timesteps - past}", list(range(past * channels)),
                         list(range(past * channels, timesteps * channels)),
                         f"Forecast all channels for the last {timesteps - past} timesteps from the first {past}; entity-held-out splits.")
                for past in (24, 16)]
    d = dataset.X.shape[1]
    return [
        TaskSpec("view_a_to_view_b", list(range(d // 2)), list(range(d // 2, d)),
                 "Predict the second authored measurement view from the first; labels remain evaluation-only."),
        TaskSpec("alternating", list(range(0, d, 2)), list(range(1, d, 2)),
                 "Predict odd-indexed measurements from even-indexed measurements across both views; fixed label-independent mask."),
    ]
