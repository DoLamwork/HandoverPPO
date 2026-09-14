# 3GPP Reference Baseline Audit

## Scope

The reference implementation is an A3-based NR handover simulator using
RSRP/SINR traces. It follows selected RRC state-machine rules from 3GPP TS
38.331, but the available traces do not contain the lower-layer information
needed for a fully conformant NR protocol implementation.

Primary reference:

- ETSI TS 138 331 V15.3.0 / 3GPP TS 38.331 Release 15:
  <https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/15.03.00_60/ts_138331v150300p.pdf>
- ETSI TS 138 133 V15.3.0 / 3GPP TS 38.133 Release 15:
  <https://www.etsi.org/deliver/etsi_ts/138100_138199/138133/15.03.00_60/ts_138133v150300p.pdf>

## Implemented definitions

### Event A3

The entering condition is:

```text
Mn + Ofn + Ocn - Hys > Mp + Ofp + Ocp + Off
```

The leaving condition is:

```text
Mn + Ofn + Ocn + Hys < Mp + Ofp + Ocp + Off
```

The code implements these inequalities and preserves the A3 state inside the
hysteresis deadband. With the current configuration, `Hys = 1 dB` and
`Off = 1 dB`, so entering requires the neighbor to be more than 2 dB above
the serving cell.

The simulator evaluates only the strongest-RSRP neighbor. 3GPP permits
measurement reporting for multiple cells; selecting the strongest candidate
is a deterministic baseline policy, not an RRC requirement.

### Time to trigger

The A3 condition must remain entered for the configured TTT. The current
40 ms value is one of the values supported by TS 38.331. TTT is configured by
the network and is not universally fixed at 40 ms.

### Radio link monitoring and RLF

TS 38.331 starts T310 after N310 consecutive lower-layer out-of-sync
indications, stops T310 after N311 consecutive in-sync indications, and
declares RLF when T310 expires.

The code now follows those counter/timer transitions. A sample between the two
thresholds generates no indication and does not advance either counter.

Important limitation: 3GPP Qin/Qout are based on hypothetical PDCCH BLER and
lower-layer evaluation periods. The dataset contains only SINR, so `-8 dB`
and `-6 dB` are explicit simulation proxies, not standardized SINR values.

### Handover execution

The selected target is latched when A3 finishes TTT. A later measurement
change cannot silently redirect an in-progress handover. Starting
reconfiguration-with-sync stops T310, as specified by TS 38.331.

The 50 ms preparation delay and 40 ms execution delay are simulation
assumptions. The target-success check, `target SINR > Qin proxy` after the
execution delay, approximates random access/T304 behavior but does not model
those procedures.

### Ping-pong

A ping-pong event is defined for this evaluation as:

```text
A -> B -> A within the configured 1 second window
```

Only successful handovers are considered. The ping-pong rate is:

```text
number of ping-pong returns / number of successful handovers
```

The one-second window is an evaluation choice, not a universal 3GPP timer.

### Failure metrics

- `HOF (%) = 100 * failed HO executions / started HO executions`
- `RLF time (%) = 100 * RLF-recovery time / simulated trace duration`
- `PP (%) = 100 * ping-pong events / successful HO executions`

RLF is intentionally not divided by the number of handovers because RLF can
occur without an active handover.

### Estimated data rate

The trace does not contain scheduled throughput. The reported rate is the
Shannon upper-bound estimate:

```text
spectral_efficiency = log2(1 + 10 ** (SINR_dB / 10))
estimated_rate_bps = bandwidth_hz * spectral_efficiency
```

The configured bandwidth is 10 MHz. During disconnection, HO execution, or
RLF recovery, the contribution is zero. This metric must be called estimated
rate or Shannon capacity, not measured user throughput.

## Remaining simulation assumptions

- RSRP/SINR samples arrive every 10 ms.
- The loader validates native sampling and does not resample traces.
- L3 filtering is applied only to RSRP for A3. Raw SINR is retained for rate
  estimation and used as the explicit RLM stream in version 1.
- L1 filtering and PDCCH BLER are not explicitly modeled.
- HO preparation signalling and target random access are represented by fixed
  delays.
- RLF recovery uses a fixed 200 ms delay before a suitable-cell search; this
  is not a complete T311/RRC re-establishment implementation.
- Cell load, scheduler allocation, MCS, HARQ, retransmissions, and protocol
  overhead are absent from the estimated data rate.

The baseline is therefore suitable as a reproducible A3 reference for the
same trace-driven environment, but it should be described as a
3GPP-inspired/A3-based simulator rather than a fully conformant NR stack.
