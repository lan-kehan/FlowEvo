# Flow-TTS Controller Discovery Implementation Spec

## 0. Goal

Implement an AutoTTS-style discovery framework for **test-time scaling controllers over flow-matching samplers**.

The system should not hand-code one fixed sampler. Instead, it should expose a constrained sampling environment and let a coding agent, such as Codex, implement candidate controllers over this environment.

The core idea is:

> A flow-matching sampler can be controlled by repeatedly moving trajectories **forward toward clean data**, optionally **previewing** intermediate clean anchors, and **backward perturbing** anchors to create local refinement or new branches. The controller decides when to spawn, move forward, preview, perturb backward, prune, and answer.

This action space should subsume existing self-refining Predict-and-Perturb sampling as a special case, while allowing more general population search and adaptive compute allocation.

---

## 1. Flow-Matching Convention

Use the following time convention throughout the code:

\[
t=0: \text{noise}, \qquad t=1: \text{clean data}
\]

A flow-matching path is:

\[
z_t = (1-t)z_0 + t z_1
\]

where:

- \(z_0 \sim \mathcal{N}(0,I)\) is the initial noise latent;
- \(z_1\) is the clean data latent;
- \(u_\theta(z_t,t,c)\) is the learned velocity field conditioned on prompt or condition \(c\).

The model predicts:

\[
u_\theta(z_t,t,c) \approx z_1-z_0
\]

From an intermediate state \(z_t\), define the clean anchor prediction:

\[
\hat z_1 = D_\theta(z_t,t,c) = z_t + (1-t)u_\theta(z_t,t,c)
\]

and the inferred noise anchor:

\[
\hat z_0 = z_t - t u_\theta(z_t,t,c)
\]

These two anchors allow deterministic forward stepping and stochastic backward perturbation to be expressed in one framework.

---

## 2. Environment-Level Action Set

The environment should expose six primitive actions:

```text
SPAWN(n)
FORWARD(particle_id, target_time, solver, cfg)
PREVIEW(particle_id, mode, scorer)
BACKWARD(anchor_id, target_time, noise_policy, num_children, mask)
PRUNE(particle_ids)
ANSWER(rule)
```

The two core generative directions are:

- `FORWARD`: move a trajectory toward the clean endpoint;
- `BACKWARD`: perturb a clean anchor back toward a chosen noise level.

The other actions are necessary controller-level operations:

- `SPAWN`: increase width;
- `PREVIEW`: pay compute to expose a clean-anchor estimate and quality signal;
- `PRUNE`: remove active particles;
- `ANSWER`: terminate and aggregate.

This mirrors the separation in AutoTTS-style environments: generation/progression actions, probe/observation actions, and pool-control/termination actions.

---

## 3. Action Semantics

### 3.1 `SPAWN(n)`

Create `n` new particles from Gaussian noise:

\[
z_0^{(j)} \sim \mathcal{N}(0,I), \quad j=1,\dots,n
\]

Each spawned particle starts at time `t = 0`.

Return:

```python
List[int]  # particle ids
```

Cost:

- No NFE cost by default.
- Actual cost starts when the particle is advanced or previewed.

---

### 3.2 `FORWARD(particle_id, target_time, solver="euler", cfg=None)`

Commit the particle state by moving it toward a larger time value.

For Euler:

\[
z_{t'} = z_t + (t'-t)u_\theta(z_t,t,c)
\]

where \(t' > t\).

The environment should support at least:

```text
euler
heun   # optional for v1
```

Return:

```python
ParticleSummary
```

Cost:

- Euler: +1 NFE
- Heun: +2 NFE

Important:

- `FORWARD` changes the current particle state.
- It does not necessarily expose a preview score.
- If the controller wants a quality signal, it must explicitly call `PREVIEW`.

---

### 3.3 `PREVIEW(particle_id, mode="clean_anchor", scorer="default")`

Expose a clean-anchor estimate from the current particle state.

For the default cheap preview:

\[
\hat z_1 = z_t + (1-t)u_\theta(z_t,t,c)
\]

The preview may be decoded and scored using a verifier or reward model:

\[
s = R(\hat z_1,c)
\]

Return:

```python
PreviewRecord(
    anchor_id: int,
    particle_id: int,
    time: float,
    z1_hat_ref: TensorRef,
    z0_hat_ref: TensorRef,
    score: Optional[float],
    score_dict: Dict[str, float],
    uncertainty: Optional[float],
    drift: Optional[float],
    embedding_ref: Optional[TensorRef],
)
```

Cost:

- `mode="clean_anchor"`: +1 NFE.
- `mode="short_rollout"`: +K NFE, where K is configured.
- `mode="full_rollout"`: cost equals the remaining integration cost to `t=1`.

The v1 implementation should support at least:

```text
clean_anchor
```

Optional modes for later:

```text
short_rollout
full_rollout
```

Important:

- `PREVIEW` is not an oracle.
- Early previews may be unreliable.
- The controller may only use preview feedback after it has explicitly called `PREVIEW`.

---

### 3.4 `BACKWARD(anchor_id, target_time, noise_policy="fresh_noise", num_children=1, mask=None)`

Starting from a clean anchor \(a=\hat z_1\), perturb backward to a selected time \(\tau\):

\[
z_\tau^{(j)} = \tau a + (1-\tau)\epsilon_j
\]

Return:

```python
List[int]  # child particle ids
```

Supported `noise_policy` values:

```text
inferred_noise  # use z0_hat from PREVIEW; deterministic / ODE-like
fresh_noise     # sample epsilon ~ N(0,I); stochastic exploration
mixed_noise     # epsilon = lambda * z0_hat + (1-lambda) * epsilon_new
masked_noise    # replace noise only in mask regions
```

For v1, implement:

```text
inferred_noise
fresh_noise
mixed_noise
```

Optional `mask`:

- If `mask is None`, perturb all latent locations.
- If `mask` is provided, perturb only selected regions and preserve the rest from the source trajectory.

Cost:

- No model NFE by itself if it only constructs latents.
- But it creates children that will later consume NFE.

Important:

- Branching is implemented by `BACKWARD(..., num_children > 1)`.
- Self-refinement is implemented by `PREVIEW` followed by `BACKWARD(..., target_time=current_time, noise_policy="fresh_noise")`.

---

### 3.5 `PRUNE(particle_ids)`

Remove particles from the active set.

Return:

```python
None
```

Cost:

- No NFE.

Important:

- Pruned particles remain in logs and histories for hindsight diagnostics.
- The controller cannot revive a pruned particle in v1.

---

### 3.6 `ANSWER(rule="best_score")`

Terminate the episode and output a final sample.

Possible aggregation rules:

```text
best_preview_score
best_final_score
score_plus_stability
latest_active
```

For v1, implement:

```text
best_preview_score
latest_active
```

Important:

- `ANSWER` is a termination action, not a generation primitive.
- If `ANSWER` requires finishing unfinished particles to `t=1`, the environment must explicitly charge the required NFE.
- Strict mode: `ANSWER` may only select among already completed particles or previously previewed anchors.

Recommended v1 behavior:

- If a chosen particle is not at `t=1`, complete it with deterministic `FORWARD` steps and charge NFE.
- Then decode and compute final score for evaluation.

---

## 4. Expressing Existing Samplers

### 4.1 Ordinary Deterministic Flow Sampling

An Euler step from \(t\) to \(t+\Delta t\) can be represented directly by `FORWARD`:

```python
FORWARD(i, target_time=t+dt, solver="euler")
```

It can also be expressed using anchor geometry:

1. `PREVIEW(i)` obtains \(\hat z_1\) and \(\hat z_0\);
2. `BACKWARD(anchor, target_time=t+dt, noise_policy="inferred_noise")` constructs:

\[
z_{t+\Delta t}=(t+\Delta t)\hat z_1 + (1-t-\Delta t)\hat z_0
\]

which expands to:

\[
z_{t+\Delta t}=z_t+\Delta t u_\theta(z_t,t,c)
\]

In code, use `FORWARD` for deterministic stepping because it is cheaper and simpler.

---

### 4.2 Self-Refining Predict-and-Perturb Sampling

Self-refining P&P is a special controller:

```python
SPAWN(1)
for t in time_grid[:-1]:
    if t in early_interval:
        for k in range(Kf):
            anchor = PREVIEW(i, mode="clean_anchor", scorer=None)
            i = BACKWARD(
                anchor_id=anchor.id,
                target_time=t,
                noise_policy="fresh_noise",
                num_children=1,
            )[0]
    FORWARD(i, target_time=next_t, solver="euler")
ANSWER(rule="latest_active")
```

This exactly matches the idea:

\[
z_t \rightarrow \hat z_1 \rightarrow z_t'
\]

where the model predicts a clean endpoint and then the endpoint is perturbed back to the same noise level.

---

### 4.3 PRISM-Like Population Search for Flow Matching

A PRISM-like controller can be expressed as:

```python
SPAWN(N)
FORWARD all particles to t_mid_1
PREVIEW all particles
PRUNE bottom particles by preview score / uncertainty
for each survivor:
    BACKWARD(anchor, target_time=t_mid_1, noise_policy="fresh_noise", num_children=b)
FORWARD all children to t_mid_2
PREVIEW all children
PRUNE to K particles
FORWARD remaining particles to t=1
ANSWER(rule="best_final_score")
```

This gives:

- width scaling through `SPAWN` and multi-child `BACKWARD`;
- signal acquisition through `PREVIEW`;
- budget allocation through `PRUNE`;
- final aggregation through `ANSWER`.

---

## 5. Runtime State

Define the state visible to controllers as:

```python
@dataclass
class ControllerState:
    prompt: str
    budget_left: int
    active_particle_ids: List[int]
    completed_particle_ids: List[int]
    pruned_particle_ids: List[int]
    particles: Dict[int, ParticleSummary]
    previews: Dict[int, PreviewRecord]
    event_log: List[EventRecord]
```

Each particle summary:

```python
@dataclass
class ParticleSummary:
    id: int
    time: float
    parent_id: Optional[int]
    source_anchor_id: Optional[int]
    nfe_used: int
    status: Literal["active", "completed", "pruned"]
    last_preview_id: Optional[int]
    num_children: int
```

Each preview record:

```python
@dataclass
class PreviewRecord:
    id: int
    particle_id: int
    time: float
    score: Optional[float]
    score_dict: Dict[str, float]
    uncertainty: Optional[float]
    drift: Optional[float]
    embedding_ref: Optional[str]
```

Important visibility rule:

> The controller can only use fields in `ControllerState`. It must not access hidden final scores, hidden latents of inactive nodes, or precomputed future outcomes.

---

## 6. Environment API

Implement a class with the following interface:

```python
class FlowTTSEnv:
    def __init__(self, model, vae, scorer, prompt, budget, time_grid, seed):
        ...

    def get_state(self) -> ControllerState:
        ...

    def spawn(self, n: int) -> List[int]:
        ...

    def forward(
        self,
        particle_id: int,
        target_time: float,
        solver: str = "euler",
        cfg: Optional[float] = None,
    ) -> ParticleSummary:
        ...

    def preview(
        self,
        particle_id: int,
        mode: str = "clean_anchor",
        scorer: Optional[str] = "default",
    ) -> PreviewRecord:
        ...

    def backward(
        self,
        anchor_id: int,
        target_time: float,
        noise_policy: str = "fresh_noise",
        num_children: int = 1,
        mask: Optional[Any] = None,
        strength: float = 1.0,
    ) -> List[int]:
        ...

    def prune(self, particle_ids: List[int]) -> None:
        ...

    def answer(self, rule: str = "best_preview_score") -> "AnswerRecord":
        ...
```

All environment actions should log structured events.

---

## 7. Controller Interface

Candidate controllers should be implemented as:

```python
class OptimalController:
    def solve(self, env: FlowTTSEnv, beta: float) -> "AnswerRecord":
        ...
```

Constraints:

1. The controller must only call public environment methods.
2. The controller must expose only one hyperparameter: `beta`.
3. Larger `beta` must monotonically correspond to larger compute budget.
4. The controller must not hardcode dataset-specific prompts, answers, seeds, or hidden metadata.
5. The controller must terminate with `env.answer(...)` before budget is exceeded.

Recommended beta mapping:

```python
def map_beta(beta: float):
    beta = float(min(max(beta, 0.0), 1.0))
    return {
        "max_particles": int(2 + 14 * beta),
        "max_preview_calls": int(2 + 12 * beta),
        "max_backward_children": int(1 + 4 * beta),
        "min_keep": int(1 + 3 * beta),
        "early_time": 0.25,
        "mid_time": 0.55,
        "late_time": 0.80,
    }
```

This is only a starting template. Codex or another explorer may edit the controller logic, but should preserve the single-`beta` design.

---

## 8. Discovery Loop

The discovery process should follow this structure:

1. Build a search environment or replay environment over a set of prompts.
2. Ask the explorer LLM to propose or edit `OptimalController`.
3. Evaluate the controller over multiple `beta` values.
4. Record:
   - final reward;
   - NFE cost;
   - preview cost;
   - action usage;
   - execution traces;
   - failure diagnostics.
5. Provide the resulting history to the explorer.
6. Repeat for multiple rounds.
7. Select the best controller on the reward-cost Pareto frontier.

---

## 9. Online Evaluation vs Offline Replay

### 9.1 Online Evaluation

Online evaluation calls the actual flow model for every action.

Pros:

- True dynamics.
- Easy to implement first.

Cons:

- Expensive for controller discovery.

### 9.2 Offline Replay

Offline replay precomputes a finite action graph for each prompt:

```text
nodes: latent states at selected times
edges: FORWARD or BACKWARD transitions
probes: PREVIEW records and scores
```

The controller is then evaluated by traversing this stored graph.

Pros:

- Cheap deterministic evaluation.
- Suitable for many Codex discovery rounds.

Cons:

- Requires discretizing the action space.
- May miss controllers that require unseen perturbations.

Recommended implementation path:

1. Implement online environment first.
2. Log all online trajectories.
3. Use those logs to create replay environments later.

---

## 10. History and Feedback Design

The discovery history is more important than the action set. The explorer needs evidence about why a controller worked or failed.

For each controller round, store:

```json
{
  "round_id": 3,
  "controller_name": "OptimalController_v3",
  "beta_sweep": [
    {"beta": 0.25, "reward": 0.61, "nfe": 42},
    {"beta": 0.50, "reward": 0.65, "nfe": 68},
    {"beta": 0.75, "reward": 0.66, "nfe": 104},
    {"beta": 1.00, "reward": 0.67, "nfe": 148}
  ],
  "action_statistics": {
    "spawn": 8.2,
    "forward_nfe_fraction": 0.72,
    "preview_nfe_fraction": 0.11,
    "backward_calls": 6.4,
    "pruned_particles": 5.7
  },
  "signal_diagnostics": {
    "preview_final_corr_by_time": {
      "0.00-0.25": 0.08,
      "0.25-0.50": 0.31,
      "0.50-0.75": 0.55,
      "0.75-1.00": 0.42
    },
    "false_prune_rate": 0.12,
    "wasted_nfe_rate": 0.34
  },
  "backward_diagnostics": {
    "fresh_noise_same_time_gain": 0.024,
    "mixed_noise_same_time_gain": 0.018,
    "late_backward_artifact_rate": 0.21
  },
  "failure_cases": [
    {
      "type": "premature_prune",
      "summary": "Low preview score but high uncertainty particle was pruned early and later hindsight showed high final reward."
    },
    {
      "type": "over_perturbation",
      "summary": "Repeated backward perturbation near late time reduced aesthetic quality."
    }
  ],
  "suggestions": [
    "Avoid pruning high-uncertainty particles before mid-stage.",
    "Use backward perturbation more in early/mid stages and less near final.",
    "Spend preview budget where preview-final correlation is highest."
  ]
}
```

The explorer should see:

- reward-NFE scaling curves;
- action usage by time bin;
- preview reliability by time bin;
- false-prune examples;
- backward perturbation ROI;
- over-perturbation cases;
- diversity collapse cases.

---

## 11. Event Logging Schema

Every action should append an event:

```python
@dataclass
class EventRecord:
    step_id: int
    action: str
    particle_ids: List[int]
    input_time: Optional[float]
    output_time: Optional[float]
    nfe_cost: int
    budget_left: int
    details: Dict[str, Any]
```

Examples:

```python
EventRecord(
    action="PREVIEW",
    particle_ids=[3],
    input_time=0.55,
    output_time=0.55,
    nfe_cost=1,
    details={"score": 0.71, "uncertainty": 0.22, "anchor_id": 19},
)
```

```python
EventRecord(
    action="BACKWARD",
    particle_ids=[7, 8, 9],
    input_time=1.0,
    output_time=0.55,
    nfe_cost=0,
    details={"anchor_id": 19, "noise_policy": "fresh_noise", "num_children": 3},
)
```

---

## 12. Metrics

Evaluate each controller with:

```text
final_reward                 higher is better
NFE                          lower is better
reward_per_NFE               higher is better
preview_calls                diagnostic
backward_calls               diagnostic
num_particles_spawned        diagnostic
false_prune_rate             diagnostic, hindsight only
wasted_nfe_rate              diagnostic, hindsight only
preview_final_correlation    diagnostic, hindsight only
```

Objective for ranking controllers:

\[
J(\pi,\beta)=\mathbb{E}[R(x_1,c)-\gamma\cdot \mathrm{NFE}]
\]

Also report the Pareto frontier over beta.

---

## 13. Minimal Baseline Controllers

### 13.1 Deterministic Baseline

```python
class DeterministicController:
    def solve(self, env, beta):
        ids = env.spawn(1)
        i = ids[0]
        for t_next in env.time_grid[1:]:
            env.forward(i, target_time=t_next, solver="euler")
        return env.answer(rule="latest_active")
```

### 13.2 Best-of-N Baseline

```python
class BestOfNController:
    def solve(self, env, beta):
        n = int(2 + 14 * beta)
        ids = env.spawn(n)
        for i in ids:
            for t_next in env.time_grid[1:]:
                env.forward(i, target_time=t_next, solver="euler")
            env.preview(i, scorer="default")
        return env.answer(rule="best_preview_score")
```

### 13.3 Self-Refine P&P Baseline

```python
class SelfRefineController:
    def solve(self, env, beta):
        Kf = int(1 + 3 * beta)
        ids = env.spawn(1)
        i = ids[0]
        for t_next in env.time_grid[1:]:
            state = env.get_state().particles[i]
            t = state.time
            if t < 0.35:  # early-stage under t=0 noise, t=1 clean convention
                for _ in range(Kf):
                    p = env.preview(i, mode="clean_anchor", scorer=None)
                    i = env.backward(
                        p.id,
                        target_time=t,
                        noise_policy="fresh_noise",
                        num_children=1,
                    )[0]
            env.forward(i, target_time=t_next, solver="euler")
        return env.answer(rule="latest_active")
```

### 13.4 PRISM-Style Flow Search Baseline

```python
class PrismStyleFlowController:
    def solve(self, env, beta):
        params = map_beta(beta)
        ids = env.spawn(params["max_particles"])

        # Warm-up to mid stage.
        for i in list(ids):
            env.forward(i, target_time=0.45, solver="euler")
            env.preview(i, mode="clean_anchor", scorer="default")

        # Select survivors.
        state = env.get_state()
        ranked = sorted(
            state.active_particle_ids,
            key=lambda pid: state.previews[state.particles[pid].last_preview_id].score,
            reverse=True,
        )
        survivors = ranked[:params["min_keep"]]
        env.prune([pid for pid in ranked if pid not in survivors])

        # Backward branch around good anchors.
        new_ids = []
        for pid in survivors:
            p = env.get_state().previews[env.get_state().particles[pid].last_preview_id]
            new_ids.extend(env.backward(
                p.id,
                target_time=0.45,
                noise_policy="fresh_noise",
                num_children=params["max_backward_children"],
            ))

        # Finish and answer.
        for i in new_ids:
            for t_next in [0.65, 0.85, 1.0]:
                env.forward(i, target_time=t_next, solver="euler")
            env.preview(i, scorer="default")

        return env.answer(rule="best_preview_score")
```

---

## 14. Explorer Prompt Template

Use the following prompt when asking Codex or another coding agent to propose a controller:

```text
You are an explorer for training-free flow-matching test-time scaling controller discovery.

Your goal is not to generate samples directly. Your goal is to implement a reusable controller over a flow-matching sampling environment.

The environment exposes six primitive actions:
1. SPAWN(n): create new trajectories from Gaussian noise.
2. FORWARD(i, target_time, solver, cfg): commit a trajectory toward the clean endpoint.
3. PREVIEW(i, mode, scorer): spend model compute to reveal a clean-anchor estimate and optional reward/verifier score.
4. BACKWARD(anchor_id, target_time, noise_policy, num_children, mask): perturb a clean anchor back to a chosen noise level; multiple children implement branching.
5. PRUNE(indices): remove active trajectories.
6. ANSWER(rule): terminate and aggregate final output.

Important rules:
- FORWARD and BACKWARD are the two core generative directions.
- PREVIEW is the only semantic observation action; preview scores may be unreliable early.
- Branching is implemented by BACKWARD with num_children > 1.
- Pruning and stopping must rely only on observed PREVIEW feedback and public state.
- Do not access hidden final rewards, hidden future trajectories, or dataset-specific shortcuts.
- The controller must expose exactly one hyperparameter beta in [0,1].
- Larger beta must monotonically use more compute.
- Optimize the reward-NFE Pareto frontier, not only a single budget point.

Implement class OptimalController with:

class OptimalController:
    def solve(self, env, beta):
        ...

Use the previous history, scaling curves, execution traces, and diagnostics to improve the controller.
```

---

## 15. Implementation Milestones

### Milestone 1: Online environment skeleton

Implement:

- particle store;
- anchor store;
- `spawn`;
- `forward` using Euler;
- `preview` using clean anchor prediction;
- `backward` with `fresh_noise` and `inferred_noise`;
- `prune`;
- `answer`;
- event logging.

### Milestone 2: Baselines

Implement:

- deterministic flow baseline;
- best-of-N baseline;
- self-refine P&P baseline;
- PRISM-style flow search baseline.

### Milestone 3: Evaluation

Implement:

- beta sweep;
- reward-NFE curve;
- event trace collection;
- action statistics;
- preview-final correlation;
- false-prune and wasted-NFE hindsight diagnostics.

### Milestone 4: Explorer loop

Implement:

- controller proposal file;
- evaluation script;
- history JSON writer;
- prompt generator for Codex;
- selection of best Pareto controller.

### Milestone 5: Offline replay

Optional but recommended after online baseline works:

- precompute finite action graphs;
- replay controller decisions without model calls;
- use replay for cheap controller search.

---

## 16. Key Design Principle

Do not make the action set too large.

The final design should be:

```text
SPAWN / FORWARD / PREVIEW / BACKWARD / PRUNE / ANSWER
```

where:

- Self-Refine is repeated `PREVIEW -> BACKWARD` at the same time;
- ordinary sampling is repeated `FORWARD`;
- branching is `BACKWARD` with multiple children;
- pruning is based on observed preview feedback;
- final selection is `ANSWER`.

This gives enough freedom for Codex to discover controllers beyond self-refinement while keeping the environment simple and implementable.

