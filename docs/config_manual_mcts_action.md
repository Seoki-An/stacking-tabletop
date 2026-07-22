# Config Manual: MCTS and Environment Action

Config files: `agent/configs/config.yml` (planning) and `agent/configs/sampling.yml` (data generation).

---

## 1. MCTS Settings (`algorithm.mcts`)

### 1.1 Tree Search Control

| Key | Type | Description |
|-----|------|-------------|
| `max_depth` | int | Maximum tree depth. Nodes beyond this depth are treated as terminal for expansion. |
| `n_iter` | int | Number of MCTS select-expand-simulate-backpropagate iterations per search call. |
| `max_search_time` | float / null | Wall-clock budget (seconds). Once elapsed, no new iterations are started. `null` or `<=0` disables the guard. **Note:** cannot interrupt an in-progress C++ `posegen`/`diffsim` call. |
| `max_children_num` | int | Maximum number of children for non-root nodes. |
| `exploration_constant` | float | PUCT exploration coefficient `c` in the selection formula `Q + c * P * sqrt(N) / (1 + n)`. Higher values favour less-visited nodes. |
| `confidence` | float | Scales the prior `P` term in PUCT. |
| `max_inference_depth` | int | Maximum depth at which the Q-function is evaluated for leaf value estimation. |
| `max_inference_step` | int | Maximum cumulative stacking steps at which Q-function inference is applied. |

Root child count is controlled by `root_proposal.keep`. Planning/test callers
detach the selected node after each step; sampler and diagnostic callers retain
the tree explicitly when they need tree-state export or inspection.

### 1.2 Root Proposal (CEM)

Controlled by `algorithm.mcts.root_proposal`. Generates the initial candidate action set at the tree root using Cross-Entropy Method (CEM) refinement.

| Key | Type | Description |
|-----|------|-------------|
| `population` | int | Number of random candidates sampled per CEM iteration. The effective root sample count is at least `environment.action.n_action_samples` so small CEM populations do not exhaust root sampling before valid later poses are reached. |
| `elite` | int | Top-K candidates selected to form the next iteration's distribution. |
| `iterations` | int | Number of CEM refinement iterations. |
| `keep` | int | Final number of CEM proposals retained as root candidates. |
| `mutation_xy_std` | float | Standard deviation (m) for XY perturbation of elite candidates. |
| `mutation_yaw_std_deg` | float | Standard deviation (degrees) for yaw perturbation of elite candidates. |

CEM only samples, mutates, scores, and deduplicates proposals. It does not run
physics. A root proposal becomes a depth-1 child and receives long simulation
before it can be expanded. Near duplicates use fixed thresholds in code
(`0.03 m` XY and `5 deg` yaw).

### 1.3 Validation (`validation`)

Every dynamic pass begins with settling simulation, which refines the candidate
pose while the existing scene is frozen. The configurable structural passes are:

| Key | Type | Description |
|-----|------|-------------|
| `validation.max_candidates` | int / null | Ranked root candidates processed per final-validation batch. |
| `validation.debug_extra_candidates` | int | Lower-ranked failures retained only for debugging after a successful batch. |
| `validation.max_place_displacement` | float / null | Optional hard displacement margin applied by long validation. |
| `validation.n_threads` | int / null | Parallel validation workers; defaults to the environment thread budget. |

Physical limits are owned by `environment.sim.short` and
`environment.sim.long`. MCTS selects the named profile explicitly; it does not
rewrite environment configuration around a simulation call. Depth-1 children
are long-simulated as a feasibility gate before deeper expansion, and promoted
descendant paths are also long-validated.
Legacy saved configs under `root_proposal.final_sim` and sibling
`algorithm.mcts.final_validation` remain readable.

### 1.4 Progressive Widening (`widening`)

Controls how quickly new children are added relative to visit count.

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable/disable progressive widening. |
| `c` | float | Widening coefficient. |
| `alpha` | float | Widening exponent. A new child is allowed when `n_children < c * N^alpha`, where `N` is the parent visit count. |

### 1.5 Action Generation (`action_generation`)

Controls batched generation of new children during node expansion.

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | int | Number of action candidates generated per expansion batch. Trades memory for expansion throughput. |
| `attempt_multiplier` | float | Multiplier on `batch_size` for the raw sampling attempt count. Defaults in code; normal configs only set it for experiments. |

The heuristic action prior is intentionally limited to posegen equilibrium
quality (`c_feq`) and placed-stone target IoU. Floor-fill, support,
orientation, grasp access, and continuation do not add prior terms.
Execution rejection remains a separate hard action mask.

### 1.6 Reward / Backpropagation (`reward`)

Controls how rewards are mixed and propagated up the tree.

| Key | Type | Description |
|-----|------|-------------|
| `mean` | float | Weight of the mean reward estimate (vs. max) when aggregating child Q-values. |
| `discount` | float | Discount factor γ applied per depth during backpropagation. |
| `optimality` | float | Interpolation weight toward the max-child Q-value (higher = more optimistic). In sampling configs this is higher (0.4) to encourage exploration. |
| `value_estimate.horizon` | int | Maximum number of unsimulated future placements represented by the mean-reward value tail. Set to `0` to disable the tail. |
| `terminal_weight` | float | Scale factor applied to terminal node rewards during backpropagation. |
| `backprop_terminal` | bool | If `True`, propagates terminal rewards all the way to the root. |

### 1.8 Sampling-Only Settings (`sampler`)

These keys appear only in `sampling.yml` and are used by the Ray episode sampler to dynamically schedule exploration.

| Key | Type | Description |
|-----|------|-------------|
| `epsilon` | float | Epsilon-greedy action noise during sampling rollouts. |
| `sampler.exploration_decay` | float | Multiplicative decay applied to the exploration constant per stacking step (`c_{k+1} = c_k * decay`). |
| `sampler.min_exploration_constant` | float | Lower bound on the decayed exploration constant. |
| `sampler.max_exploration_constant` | float | Upper bound (clamp) on the exploration constant. |

---

## 2. Environment Action Settings (`environment.action`)

### 2.1 General

| Key | Type | Description |
|-----|------|-------------|
| `banned_stone_ids` | list[int] | Stone model IDs (by number in `model_<id>.obj`) excluded from action selection. They remain in the scene as obstacles. Example: `[2, 11]`. |
| `pose_from_scan` | bool | If `True`, uses scanned pick-pose orientations instead of synthetic rotation grids. Requires `pose_data_path` to be valid. |
| `pose_data_path` | str | Path to the `.pkl` file containing pre-scanned pick poses. |
| `n_action_samples` | int | Total number of action candidates generated per call to `get_action_samples()`. |
| `max_pose_per_stone` | int | Maximum pose candidates allocated per individual stone. |
### 2.2 Rotation Grid (`action.rotation`)

Used when `pose_from_scan: False` to enumerate candidate stone orientations.

| Key | Type | Description |
|-----|------|-------------|
| `angles_x` | list[float] | X-axis tilt angles (degrees) to try for each stone. `[0, 90]` means flat and tipped. |
| `angles_z` | list[float] | Z-axis yaw angles (degrees). `[0, 90, 180, 270]` gives 4 yaw orientations. |

### 2.3 Support Constraint (`action.support_constraint`)

Filters candidate action placements that lack sufficient support from already-placed stones or the ground.

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable/disable support constraint evaluation. |
| `hard_filter` | bool | If `True`, post-pose candidates that fail the geometric support constraint get `-inf` before root validation. Default is soft: record the failure but let long simulation decide. |
| `min_supports` | int | Minimum number of support sources (placed stones + ground) required. |
| `large_below_volume_ratio` | float | Ratio threshold for requiring a larger stone to be placed below a smaller one. |
| `connected_ground.enabled` | bool | Enable the MCTS isolated-ground guard that rejects late ground-only placements disconnected from existing stones. |
| `connected_ground.allow_lower_fill` | bool | Let candidates marked `lower_fill_candidate` bypass the isolated-ground guard so floor-fill sampling can place into low empty cells. |
| `xy_factor` | float | Scales the candidate stone's XY footprint radius when searching for nearby support stones. |
| `z_tolerance` | float | (m) Vertical search window around the candidate stone's base to find support stones. |
| `pair_z_tolerance` | float | (m) Looser vertical window used for support-pair checks. |
| `pair_distance_scale` | float | Scales the inter-stone distance threshold for valid support pairs. |
| `pre_pose_filter` | bool | If `True`, apply support filtering before posegen (on initial planar pose), not just after. |
| `pre_pose_activate_after` | int | Only activate `pre_pose_filter` after this many stones have been placed. |
| `pre_pose_xy_factor` | float | XY footprint scale used during the pre-pose (pre-posegen) support check. |
| `pre_pose_fallback_to_unfiltered` | bool | If all pre-pose candidates are filtered, allow the unfiltered set through instead of returning empty. |
| `pre_pose_allow_empty_space` | bool | Allow pre-pose anchors that fall in visually empty (unfilled) target-wall space even if support is marginal. |
| `empty_space_distance_scale` | float | Scale on the stone radius used to define "empty space" for the above allowance. |

### 2.4 Planar Pose Sampling (`action.planar`)

Controls how XY anchor positions are scored and selected before posegen optimization.

| Key | Type | Description |
|-----|------|-------------|
| `variance` | float | (m) Gaussian noise added to sampled XY anchors. `0.0` disables jitter. |
| `score_model` | str | Scoring strategy for XY anchors. Options: `"score"` (active default), `"heuristic"`, `"cnn"`. |
| `score_map_size` | list[int] | `[W, H]` resolution of the internal score map used by score-based sampling. Coarser = faster. |
| `boltzmann_temperature` | float | Temperature for Boltzmann sampling over the score map. Lower values make sampling greedier. |
| `boltzmann_min_probability` | float | Minimum probability floor for each score-map cell before sampling. |
| `target_mask_margin` | float | (m) Margin added around the target-wall footprint mask when filtering anchors. |
| `place_offset` | float | (m) Vertical offset added to the computed placement height. |

#### Score Weights (`planar.score`)

Active when `score_model: "score"`. Samples XY anchors from a target-wall score
grid. The score is exponentiated with the same Boltzmann sampler used by other
planar score modes. Active-floor connectedness is a score term unless
`u_shape_min_frontier_contact_cells > 0`; U-shape rejection is controlled by
`u_shape_filter`.

| Key | Type | Description |
|-----|------|-------------|
| `height` | float | Weight for normalized `h - h_min`. Negative values prefer lower floor regions. |
| `connectedness` | float | Weight for candidate footprint contact with existing active-floor stones. |
| `open_area` | float | Weight for how much of the candidate footprint fills currently empty active-floor cells. |
| `fill_area` | float | Weight for absolute active-floor empty-cell coverage, normalized within the candidate pool. |
| `frontier` | float | Weight for frontier-contact density around the candidate footprint. |
| `target_boundary` | float | Weight for target-boundary proximity of the stone-specific valid anchor at its support height. The term is `0` at the target center and `1` at a valid side or corner after taper and stone inset are applied. Anchors outside that local tapered boundary receive no bonus. |
| `excavator_distance` | float | Tie-break weight that prefers XY anchors farther from the excavator-base planner origin when floor level is otherwise comparable. |
| `excavator_distance_axis` | str | Axis used by the excavator-distance term: `"xy"` uses Euclidean XY distance, `"x"` uses only `excavator_xy[0]`, and `"y"` uses only `excavator_xy[1]`. |
| `excavator_xy` | list[float] / null | Excavator-base XY in planner-local target coordinates. `generate_sequence.py` sets this to `-target_structure_offset` so global excavator origin and local sampled XY are compared in the same frame. |
| `elite_fraction` | float | Fraction of each per-stone XY sample budget filled deterministically from highest-score cells before Boltzmann sampling fills the remainder. |

#### Floor-Fill Layer Constraints (`planar.floor_fill`)

Controls active-floor detection and geometric constraints used by the score
sampler.

| Key | Type | Description |
|-----|------|-------------|
| `connected_after` | int | Apply the active-floor connectedness hard gate after this many stones have been placed, when `u_shape_min_frontier_contact_cells > 0`. |
| `u_shape_filter` | bool | If `True`, reject floor-fill anchors whose estimated footprint creates an enclosed or narrow-mouth empty pocket on the active floor. |
| `u_shape_footprint_scale` | float | Scale applied to the candidate stone XY extent when estimating its grid footprint for the U-shape filter. |
| `u_shape_require_frontier_contact` | bool | If `True`, reject active-floor anchors whose estimated footprint does not touch the occupied active-floor frontier. Keep disabled to let low disconnected floor cells remain scoreable. |
| `u_shape_min_frontier_contact_cells` | int | Minimum estimated footprint cells that must touch the active-floor frontier. `0` disables the connectedness hard gate and makes connectedness only a score term. |
| `u_shape_min_empty_cells` | int | Minimum empty pocket size, in active-floor occupancy cells, before the U-shape filter can reject an anchor. |
| `u_shape_min_contact_cells` | int | Minimum occupied-neighbor contact around an empty component before treating it as a pocket. |
| `u_shape_max_opening_ratio` | float | Maximum `target-boundary opening cells / occupied-contact cells` allowed for a candidate-created pocket. Higher is stricter. |
| `lower_floor_fill_reject_stacked` | bool | Optional hard rejection for post-posegen candidates whose solved pose is above the active floor while active-floor occupancy is below `lower_floor_fill_ratio`. Keep disabled unless diagnosing upper-course over-selection because it can starve MCTS candidates. |
| `boundary_inset_radius_scale` | float | Insets boundary anchors inward by `stone_radius * scale` to avoid edge overhang. |

#### Floor-Fill Orientation Filter (`planar.floor_fill.orientation`)

Post-posegen mask applied to boundary/corner floor_fill placements to ensure the top-exposed surface normal faces inward toward the target structure.

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable/disable the orientation filter. |
| `boundary_margin_radius_scale` | float | Stones whose XY anchor is within `stone_radius * scale` of the target boundary are subject to the filter. |
| `offsets_deg` | list[float] | Yaw offset candidates (degrees) tried when biasing the initial orientation toward the target centre. |
| `upper_face_min_z` | float | Minimum Z component of the face normal to count as "upper-facing". |
| `upper_face_min_horizontal` | float | Minimum horizontal (XY) magnitude of the face normal to count as "horizontal enough to be inward". |
| `min_upward_dot` | float | The area-weighted mean upper-face normal must have at least this dot product with `+Z`. Rejects near-vertical placements. `0.7` ≈ 45° from vertical (from Johns et al. 2020). |
| `max_inward_angle_deg` | float | Maximum angle (degrees) between the mean horizontal face normal and the inward XY direction. `90` = nonneg dot product (loose). Lower for stricter alignment. |

#### CNN Score Model (`planar.cnn`)

Used when `score_model: "cnn"`. Loads a `HeightmapValueModel` to produce a learned score map.

| Key | Type | Description |
|-----|------|-------------|
| `config` | str | Path to the heightmap value model config YAML. |
| `weights` | str | Path to the model state-dict (`.pkl` or `.pt`). Empty string = untrained (random weights). |

---

## 3. Reward Settings (`environment.reward`)

### 3.1 Reward Weights (`reward.weights`)

| Key | Type | Description |
|-----|------|-------------|
| `stability` | float | Weight for the physics-simulation stability reward (stones do not move after placement). |
| `stone_IoU` | float | Weight for IoU between the placed stone's volume and the target wall volume. |
| `target_IoU` | float | Weight for the cumulative scene-vs-target fill ratio. Higher values dominate the reward signal. |
| `place_stability` | float | Weight for the perturbed-placement stability reward (stone survives small position/rotation noise). |
| `large_stone_lower` | float | Weight penalising placements that put a large stone above a smaller one. |
| `inward_orientation` | float | Weight rewarding placements whose top-exposed normal points inward toward the target. |
| `support` | float | Weight rewarding placements with sufficient support from neighbouring stones/ground. |

### 3.2 Reward Thresholds

| Key | Type | Description |
|-----|------|-------------|
| `vel_integral_thresh` | float | Velocity-integral threshold above which a placement is marked as failed/unstable. |
| `posegen_thresh` | float | Maximum posegen constraint residual before the optimised pose is rejected. |
| `IoU_thresh` | float | Minimum target IoU for an episode to count as successful. |

### 3.3 Place Stability (`reward.place_stability`)

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable/disable the perturbed-placement stability check. |
| `n_noise` | int | Number of noise samples used to estimate stability. More samples = slower but more accurate. |
| `position_std` | list[float] | `[x, y, z]` standard deviations (m) of the position noise. |
| `rotation_std_deg` | float | Standard deviation (degrees) of the rotation noise. |
| `rotation_weight` | float | (m/rad) Converts rotation perturbation to an effective displacement for the stability metric. `0.0` = translation-only. |
| `seed` | int | RNG seed for reproducible noise samples. |

### 3.4 Support Reward (`reward.support`)

| Key | Type | Description |
|-----|------|-------------|
| `desired_sources` | int | Normalisation cap; `support_count / desired_sources` (clipped to 1). |
| `ground_z` | float | (m) Placements whose bottom is at this height are counted as having ground support. |

---

## 4. Simulation Settings (`environment.sim`)

| Key | Type | Description |
|-----|------|-------------|
| `dt` | float | (s) Physics timestep per diffsim step. |
| `max_t` | float | (s) Maximum simulation time before a placement is declared settled or failed. |
| `min_t` | float | (s) Minimum simulation time before early-stop energy/velocity criteria are checked. |
| `extra_n_step` | int | Additional steps run after the energy threshold is first met. |
| `energy_thresh` | float | Kinetic energy threshold for early-stop (stone has essentially come to rest). |

---

## 5. Height Map Settings (`environment.height_map`)

| Key | Type | Description |
|-----|------|-------------|
| `resolution` | list[int] | `[W, H]` pixel dimensions of the orthographic depth render. `128x128` is the current default. |
| `margin` | float | (m) Padding around the target wall bounding box in the ortho camera frame. Larger values show more context. |

---

## 6. Target Wall Settings (`environment.target`)

| Key | Type | Description |
|-----|------|-------------|
| `width` | float | (m) Target wall width. |
| `length` | float | (m) Target wall length. |
| `height` | float | (m) Target wall height. |
| `taper` | float | (degrees) Wall taper angle (narrowing toward the top). |
| `origin` | list[float] | `[x, y]` position of the target wall centre in the world frame. |
| `randomize` | bool | If `True`, target wall dimensions and position are randomised each episode (used during data generation). |

---

## 7. Quick Reference: Key Differences Between `config.yml` and `sampling.yml`

| Parameter | `config.yml` (planning) | `sampling.yml` (data gen) |
|-----------|-------------------------|---------------------------|
| `n_iter` | 100 | 128 |
| `max_search_time` | 600 s | 300 s |
| `root_proposal.keep` | 24 | 16 |
| `root_proposal.iterations` | 3 | 3 |
| `action_generation.batch_size` | 32 | 8 |
| `reward.optimality` | 0.1 | 0.4 |
| `exploration_constant` | 0.5 | 1.0 |
| `environment.n_stone` | 20 | 16 |
| `environment.data.load_dir` | `assets/stone` | `assets/stone_synthetic` |
| `environment.target.randomize` | False | True |
| `environment.height_map.margin` | 1.0 m | 0.5 m |
| `reward.weights.place_stability` | 0.5 | 0.0 |
| `reward.weights.large_stone_lower` | 2.0 | 0.0 |
