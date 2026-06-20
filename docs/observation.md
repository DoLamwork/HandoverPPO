# A3 Decision Observation

`A3ObservationBuilder` produces a causal 19-element `float32` vector only
after A3 has remained valid for TTT. It is updated exclusively through past
`TickResult` objects and has no access to future trace samples.

| Feature | Range | Purpose |
|---|---:|---|
| serving spectral efficiency | `[0, 1]` | Current serving-link utility |
| target spectral efficiency | `[0, 1]` | Expected target-link utility |
| delta spectral efficiency | `[-1, 1]` | Immediate rate advantage of HO |
| A3 margin now/mean/min | `[-1, 1]` | Strength and persistence of A3 |
| A3 margin slope | `[-1, 1]` | Whether target advantage is increasing |
| A3 margin standard deviation | `[0, 1]` | Radio instability during TTT |
| serving/target SINR slopes | `[-1, 1]` | Short-term link direction |
| serving gap to Qout | `[-1, 1]` | Risk of future out-of-sync |
| target gap to Qin | `[-1, 1]` | Likelihood that target is usable |
| N310 progress | `[0, 1]` | Accumulated out-of-sync risk |
| T310 active/progress | `[0, 1]` | Immediacy of RLF |
| serving residence time | `[0, 1]` | Stability on the current cell |
| time since last HO | `[0, 1]` | Recency of mobility action |
| target is previous cell | `{0, 1}` | Direct A-B-A ping-pong warning |
| recent HO count | `[0, 1]` | General short-term mobility churn |

A3 margins are derived from the filtered `rsrp_a3_db` stream. Spectral
efficiencies and SINR slopes use `sinr_rate_db`; Qin/Qout gaps use
`sinr_rlm_db`. Spectral efficiencies are clipped using the protocol
configuration before normalization. Slopes are least-squares estimates in
dB/s. Time and count values saturate at documented configuration limits.

The feature names, bounds, and mapping used for logs are exposed directly:

```python
builder.feature_names
builder.low
builder.high
builder.as_dict(observation)
```
