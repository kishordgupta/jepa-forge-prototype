# Ten additional synthetic dataset families

The expanded suite adds **ten distinct generation mechanisms**, comprising six
forecasting datasets and four tabular classification datasets. These are authored
software benchmarks for exercising task compilation and representation learning.
They are not observations, calibrated scientific simulators, evidence of physical
validity, or demonstrations of real-world effectiveness. The Lorenz and
Mackey–Glass families adapt named mathematical systems; the other eight are
explicitly authored mechanisms described below.

Implementation: [`synthetic_datasets.py`](../src/jepa_forge/synthetic_datasets.py).
Tests: [`test_synthetic_datasets.py`](../tests/test_synthetic_datasets.py).
The implementation is the authoritative numerical definition, and each
`RawDataset.metadata` also records the family, version, seed, parameters, source,
feature layout, label boundary, and split policy. All generated values are
designated CC0-1.0. No external data files or network access are needed.

## Shared contracts and dataset inventory

`load_synthetic_dataset(name, seed=2026)` uses a local NumPy random generator.
Different families have different code and equations; they are not copies of one
generator with different seeds. Determinism means repeated calls with the same
software and seed produce identical arrays. It does not promise bitwise equality
across all future numerical-library versions.

Every family contains **2,000 rows**. All features are finite numerical
measurements, with no appended label, entity identifier, or split identifier.
Classification labels reside solely in `RawDataset.y` with the role
`evaluation_only`. The self-supervised backend receives no classification labels.
Generated columns can still correlate with a label, as required for a meaningful
classification evaluation.

| Name | Mechanism | Input layout | Flattened features | Evaluation |
|---|---|---:|---:|---|
| `synthetic_lorenz` | Three-state nonlinear flow | 32 × 3 | 96 | Forecasting |
| `synthetic_mackey_glass` | Delayed nonlinear feedback | 32 × 3 | 96 | Forecasting |
| `synthetic_switching_var` | Three-regime vector autoregression | 32 × 4 | 128 | Forecasting |
| `synthetic_chirp_seasonal` | Frequency drift, seasonality and trend | 32 × 3 | 96 | Forecasting |
| `synthetic_coupled_oscillators` | Coupled driven cubic oscillators | 32 × 4 | 128 | Forecasting |
| `synthetic_nonlinear_ar` | Nonlinear cross-lag autoregression | 32 × 3 | 96 | Forecasting |
| `synthetic_nonlinear_multiview` | Shared latent factors, different nonlinear views | 2 × 32 | 64 | Three classes |
| `synthetic_hierarchical_multiclass` | Six subclasses within three parents | 2 × 40 | 80 | Six classes |
| `synthetic_sparse_interactions` | Pairwise interactions among nuisance features | 2 × 48 | 96 | Two classes |
| `synthetic_manifold_nuisance` | Rolled manifold with independent nuisance views | 2 × 48 | 96 | Six classes |

All forecast datasets contain **100 independent entities**, each with a retained
trajectory of 184 timesteps. A 32-timestep window with stride 8 produces
20 overlapping windows per entity. Windows are flattened in time-major order,
and `feature_time_indices` records the timestep for every column. Entity IDs are
split metadata, not features. Inclusive intervals `[start, start + 31]` describe
the raw retained coverage of each window. A 60/20/20 entity split gives
60/20/20 entities and 1,200/400/400 rows; overlapping windows from an entity remain
in its one partition. A row-random split would be inappropriate for these series.

The four tabular datasets contain independent draws from a fixed authored
distribution. They use the existing label-independent 60/20/20 row split, also
giving 1,200/400/400 rows. They have no entity-group or interval fields.

`propose_synthetic_tasks(dataset)` returns exactly two predetermined policies:

- Series: `past24_future8`, then `past16_future16`. All channels in the context
  precede every target timestep. The horizons are 8 and 16 retained timesteps.
- Tabular: `view_a_to_view_b`, then `alternating`. The first predicts the entire
  second measurement view from the first; the second predicts odd-indexed
  features from even-indexed features across both views.

Policies do not inspect labels or held-out values. Compare methods **within the
same dataset and policy**. The two series policies have different history and
horizon budgets. The two tabular policies reveal different measurements. A
cross-policy score difference does not by itself identify a better task compiler.

## Forecast generation mechanisms

All noise notation below denotes independent Gaussian draws unless a recursion
explicitly introduces dependence. Time and signal units are arbitrary synthetic
units. Entity parameters are sampled independently once per entity. Measurements
are generated once per trajectory, then shared consistently by overlapping
windows; a reused timestep is not assigned new noise in each window.

### 1. Lorenz

For each entity, integrate

```text
dx/dt = 10 (y - x)
dy/dt = x (rho - z) - y
dz/dt = x y - (8/3) z
```

using classical fourth-order Runge–Kutta, step size `0.02`, and 500 discarded
steps. Sample `rho ~ Uniform(26,30)`, with initial `x,y ~ Uniform(-12,12)` and
`z ~ Uniform(8,28)`. The three channels are the resulting `x,y,z` plus measurement
noise with standard deviation `0.05`.

The underlying equations are from
[Lorenz, *Deterministic Nonperiodic Flow* (1963)](https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml).
The parameter distribution, solver, noise, and windows are pilot choices, not a
reproduction of the original numerical experiment. No Lyapunov calculation is
performed, so the implementation does not certify every sampled trajectory as
chaotic. It introduces nonlinear state coupling and sensitive trajectory
dependence beyond the original two-channel sinusoidal sensor benchmark.

### 2. Mackey–Glass-inspired delayed dynamics

Use the explicit discrete update

```text
x[t] = max(0.01,
           (1-gamma) x[t-1]
           + beta x[t-1-delay] / (1+x[t-1-delay]^10)
           + Normal(0,0.002))
```

with `beta ~ Uniform(0.18,0.23)`, `gamma ~ Uniform(0.08,0.12)`, and an integer
delay sampled uniformly from 14 through 22. The initial 23 history values have
an entity baseline `Uniform(0.8,1.3)` plus noise of standard deviation `0.02`.
Discard 400 recurrence steps and retain an additional eight-step buffer before
the exported trajectory. Channels are `x[t]`, `x[t-4]`, and
`x[t]^2 + 0.2 x[t-8]`, each with measurement noise of standard deviation `0.005`.
All observations are causal.

The delayed feedback form is inspired by
[Mackey and Glass, *Oscillation and Chaos in Physiological Control Systems* (1977)](https://doi.org/10.1126/science.267326).
This uses an authored `dt=1` Euler-style discrete recurrence, positivity floor,
process noise, variable parameters, and measurement functions. It is **not a
validated continuous delay-differential-equation solver** and is not a
physiological model. The long delay and nonlinear sensor transform distinguish
it from the short-memory sinusoidal original benchmark.

### 3. Switching vector autoregression

Use a four-dimensional state with three hidden regimes:

```text
x[t] = A[regime[t]] x[t-1] + b[regime[t]] + scale * Normal(0,0.11)
```

An entity-specific switch probability `p ~ Uniform(0.025,0.09)` controls each
step. With probability `p`, switch uniformly to one of the other two regimes;
otherwise retain the regime. `scale ~ Uniform(0.8,1.3)`. Start from independent
`Normal(0,0.3)` states and a uniform regime, discard 100 steps, and add independent
measurement noise of standard deviation `0.025`. Regime states are hidden and
are not exported as inputs or labels.

```text
A0 = [[ .70, .15,   0,   0],     b0 = [ .10,-.08, .05,   0]
      [-.10, .65, .10,   0],
      [   0,-.15, .60, .15],
      [ .10,   0,   0, .65]]
A1 = [[ .40,-.35,   0, .10],     b1 = [-.12, .08,   0, .10]
      [ .30, .35, .10,   0],
      [ .10,   0, .50,-.25],
      [   0, .10, .25, .45]]
A2 = [[-.45, .15,   0,   0],     b2 = [   0,   0,-.10,-.08]
      [ .10,-.40, .20,   0],
      [   0, .10,-.35, .20],
      [ .15,   0, .10,-.40]]
```

Regime switches alter both persistence and cross-channel dependence. The
absolute row sums of these matrices are below one, giving a bounded deterministic
linear update for bounded inputs; Gaussian innovations themselves have unbounded
support. No application-specific regime interpretation is claimed.

### 4. Chirp, seasonality and trend

Let `t=0,...,183`, phase `phi ~ Uniform(-pi,pi)`, initial frequency
`f0 ~ Uniform(0.012,0.035)`, frequency drift `d ~ Uniform(0.00004,0.00018)`,
amplitude `a ~ Uniform(0.7,1.5)`, and slope `m ~ Uniform(-0.004,0.004)`:

```text
theta[t] = 2*pi*(f0*t + 0.5*d*t^2) + phi
envelope[t] = a*(1 + 0.25*sin(2*pi*t/90 + phi))
season[t] = 0.35*sin(2*pi*t/18 + phi/2)
trend[t] = m*(t-92)
noise[t] = 0.75*noise[t-1] + Normal(0,0.04), noise[0]=0
channel0 = envelope*sin(theta) + season + trend + noise
channel1 = 0.8*envelope*cos(theta+0.3) - 0.5*season + 0.7*trend
channel2 = 0.5*envelope*sin(2*theta) + 0.8*season - 0.4*trend + 0.5*noise
```

Add measurement noise of standard deviation `0.03` to all three channels. This
adds drifting frequency, multiple harmonics, slow modulation and trend; the
original sensor benchmark has a fixed entity frequency.

### 5. Coupled cubic oscillators

For two oscillator positions `q_i` and velocities `v_i`, with the other
oscillator indexed by `j`, use

```text
acceleration_i = -omega_i^2*q_i - .12*q_i^3 - .10*v_i
                 + coupling*(q_j-q_i)
                 + .35*sin(drive_frequency*t+phase_i)
v_i_next = v_i + dt*acceleration_i + Normal(0,.008)
q_i_next = q_i + dt*v_i_next
```

Here `dt=0.08`, `omega_i ~ Uniform(0.7,1.2)`, coupling is
`Uniform(0.12,0.35)`, and the entity's common drive frequency is
`Uniform(0.5,0.9)`. Each oscillator gets its own `Uniform(-pi,pi)` drive phase.
Initial positions and velocities have standard deviations `0.7` and `0.2`.
After 250 discarded steps, record channels `[q0,q1,v0,v1]` with measurement noise
standard deviation `0.015`. This is semi-implicit Euler with an explicitly
specified discrete velocity innovation, not a calibrated stochastic differential
equation solver. Nonlinear restoring forces, bidirectional coupling and explicit
velocity channels make this structurally different from a sampled sinusoid.

### 6. Nonlinear autoregression

With three initial lagged vectors drawn from `Normal(0,0.3)`, define

```text
f_x = .60*x[t-1] + .40*tanh(y[t-2]*z[t-1])
f_y = .45*y[t-1] + .35*sin(x[t-1]) - .15*z[t-3]
f_z = .50*z[t-1] + .30*tanh(x[t-1]-y[t-2])
state[t] = gain*f + entity_bias + sigma(state[t-1])*Normal(0,1)
sigma(previous_state) = .06 + .06*abs(tanh(previous_state))
```

The entity gain is `Uniform(0.75,1.15)` and its three biases are
`Normal(0,0.10)`. Discard 200 computed timesteps, then add measurement noise
standard deviation `0.01`. Multiplicative cross-lag terms, three lag lengths,
and heteroscedastic innovations introduce different prediction demands from
the original stationary AR(1) noise component.

## Tabular generation mechanisms

Projection matrices below contain independent standard-normal entries normalized
to unit Euclidean length per column. They are drawn once per dataset and shared
by its rows. Such shared measurement functions do not couple independent rows.
All four families use two equally sized views, with no exact feature-copy
construction between views. Separate measurement noise prevents accidental
duplicate observations in the tested seeds.

### 7. Nonlinear multiview factors

Draw six standard-normal latent factors per row. Each view has 32 features:

```text
view_a = tanh(z @ A) + .15*z0*z1*w_a + Normal(0,.08)
view_b = sin(z @ B) + .20*(z2^2-z3^2)*w_b + Normal(0,.08)
```

`w_a,w_b` are independent standard-normal row vectors. The three evaluation
classes are the argmax of scores
`[z0+.8*z1*z2, -z0+.6*sin(z3), z4-.5*z5+.4*z1^2]`. This tests shared information
across differently warped views with interaction terms.

### 8. Hierarchical multiclass mixture

Draw one of six subclasses uniformly; subclasses are paired under three parent
classes. Parent centers sit on a radius-2.2 circle in the first two coordinates
of an eight-dimensional latent space. Each parent has a random unit child
direction. Its two children have offsets `-0.9` and `+0.9` along that direction.
Add latent noise standard deviation `0.45`:

```text
z = parent_center[y//2] + (2*(y%2)-1)*.9*child_direction[y//2] + noise
view_a = z @ A + Normal(0,.12)
view_b = tanh(z @ B) + .12*z2*z3*w + Normal(0,.12)
```

Each view has 40 features. Labels specify mixture membership during generation
but are never appended to the measured features. The evaluation asks for the six
subclasses, which include both broad parent separation and finer child structure.

### 9. Sparse interactions with nuisance features

Draw ten standard-normal latent factors. View A contains these ten factors with
noise standard deviation `0.04`, followed by 38 independent standard-normal
nuisance features. View B contains 16 pairwise products with noise standard
deviation `0.06`, followed by 32 independent standard-normal nuisance features.
Product pairs are `(i,(i+1) mod 10)` for `i=0,...,9` plus `(i,i+4)` for `i=0,...,5`.
The binary evaluation label is

```text
1[z0*z1 + .8*z2*z3 - .5*z4 + .35*sin(z5) + Normal(0,.1) > 0].
```

There are 96 measured features, including 70 nuisance features. View B also
contains noise that cannot be predicted from view A; a successful task generator
must not imply every target dimension is inferable.

### 10. Manifold with independent nuisance views

Draw `theta ~ Uniform(1.5*pi,4.5*pi)` and `height ~ Uniform(-1,1)`. The first
view's signal is a 16-dimensional projection of
`[theta*cos(theta)/8, height, theta*sin(theta)/8]`. The second view's signal is
`tanh` applied to a 16-dimensional projection of
`[sin(theta),cos(theta),height,height*sin(theta)]`. Each signal gets noise
standard deviation `0.08`.

Each view also contains 32 nuisance measurements generated from its own four
independent standard-normal nuisance factors, projected with scale `2.5` and
noise standard deviation `0.2`. The views' nuisance factors are independent of
each other, of the intrinsic manifold, and of the label. The label is
`2*floor((theta-1.5*pi)/pi) + 1[height>0]`, giving six classes. The second signal
view is periodic in `theta`, so it can lose distinctions preserved by the rolled
first view. High raw nuisance variance is removed by per-column normalization,
but the 64 nuisance dimensions and their within-view correlation remain.

## Seed-2026 audit observations

These are observed raw minima and maxima over generated arrays, rounded to
three decimals; they are **not theoretical bounds**. Gaussian noise has
unbounded support. Ranges are descriptive checks and do not select policies,
fit normalizers, or tune models.

| Family suffix | Observed minimum | Observed maximum | Class counts in ID order |
|---|---:|---:|---|
| `lorenz` | -27.066 | 48.360 | — |
| `mackey_glass` | 0.141 | 3.143 | — |
| `switching_var` | -0.894 | 0.928 | — |
| `chirp_seasonal` | -2.175 | 2.337 | — |
| `coupled_oscillators` | -2.404 | 2.485 | — |
| `nonlinear_ar` | -1.416 | 0.983 | — |
| `nonlinear_multiview` | -7.109 | 5.957 | 636, 624, 740 |
| `hierarchical_multiclass` | -3.357 | 3.255 | 320, 323, 337, 340, 334, 346 |
| `sparse_interactions` | -7.726 | 8.812 | 1,016, 984 |
| `manifold_nuisance` | -11.513 | 11.291 | 344, 309, 339, 319, 333, 356 |

The automated tests check reproducibility and changed-seed variation; finite,
nonduplicate rows; distinct generator descriptions and arrays; both policies for
every family passing the compiler; no entity crossing between splits; strictly
past-to-future feature indices; labels absent from input roles; and representative
one-epoch CPU training for a series and a tabular dataset. They do not establish
scientific fidelity or forecast performance. GPU experiment outcomes must be
read from the separate recorded experiment artifacts.
