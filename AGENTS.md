# Multi-agent research guidance

## Research objective

Study and develop reproducible evidence for PPO-based handover optimization for UAVs in 5G/6G networks, with an ISAC (integrated sensing and communication) extension. Distinguish implemented behaviour from proposals or assumptions.

## Project boundaries

- `data_generator/` produces mobility and radio traces only; it does not choose a serving cell or implement handover decisions.
- Handover, PPO/SMDP, evaluation, and 3GPP baseline code live under `src/ho_optim_drl/`.
- Treat `data/` and existing `results/` as research records. Do not overwrite them; place new attributable outputs in a new subdirectory.
- Do not alter simulator source code unless the user explicitly requests it.
- Do not claim 3GPP, 5G/6G, or ISAC compliance without a cited primary source and an explicit implementation-gap account.

## Coordination

The lead agent owns decomposition, integration, decisions, and final repository changes. Use the specialist profiles in `.codex/agents/` for focused work. Specialists must stay within their profile boundaries and must not silently expand scope.

Before modifying tracked files, inspect the working tree and preserve unrelated user changes. Prefer small, reviewable patches. Never modify simulation source code as a side effect of research, review, or experimentation.

## Evidence and reporting contract

Every specialist report must include:

1. `Scope`: question addressed and files, data, or sources inspected.
2. `Findings`: concise numbered claims; separate observations from inferences.
3. `Evidence`: source URLs/DOIs or exact repository paths and commands; state version, dataset split, seed, checkpoint, and configuration where relevant.
4. `Assumptions and limits`: unresolved items, validity threats, and what was not checked.
5. `Handoff`: recommended next action and files created or changed.

Use SI units, name reward terms and signs explicitly, report per-speed as well as aggregate UAV results, and compare against the project 3GPP baseline on the same split. An ISAC proposal must state sensing observables, latency/accuracy assumptions, their availability to the policy, and communication trade-offs.

## Verification

Run the narrowest relevant checks. For Python changes, at minimum run `python -m py_compile` for touched modules; for experiments, record the exact command and verify output metadata. Never present an unrun command as a result.
