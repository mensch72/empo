# Design Plan Status

This index tracks the current status of the design documents under `docs/plans/`.
GitHub issues and pull requests are the source of truth for ongoing work; some
older plan documents still say "Proposed" or "Planning" even when substantial
implementation has already landed.

| Plan | Status | Tracking / landed work | Notes |
| --- | --- | --- | --- |
| [`bwind_parallel.md`](./bwind_parallel.md) | In progress | Related issue: [#17](https://github.com/mensch72/empo/issues/17) | The plan is still relevant, but the proposed backward-induction parallelization refactor is not fully landed. |
| [`curiosity.md`](./curiosity.md) | In progress | Landed PR: [#63](https://github.com/mensch72/empo/pull/63) | Curiosity support exists in code (`count_based_curiosity.py`, `rnd.py`), but the document still describes additional exploration work. |
| [`dueling_q_r_architecture.md`](./dueling_q_r_architecture.md) | Deferred | — | No corresponding dueling-architecture implementation is present in the current Phase 2 code. |
| [`hierarchical_world_models.md`](./hierarchical_world_models.md) | In progress | Landed PRs: [#128](https://github.com/mensch72/empo/pull/128), [#129](https://github.com/mensch72/empo/pull/129) | The document itself says "In Progress (Tasks 1–11 complete)", and substantial hierarchical infrastructure has landed under `src/empo/hierarchical/`. |
| [`learning_qr_scale.md`](./learning_qr_scale.md) | Implemented | Landed PR: [#71](https://github.com/mensch72/empo/pull/71) | The value-transform machinery described by the plan is present in `src/empo/learning_based/phase2/value_transforms.py`. |
| [`llm2model.md`](./llm2model.md) | Deferred | Related issue: [#150](https://github.com/mensch72/empo/issues/150) | There is exploratory LLM-modeling work, but the end-to-end world-model generation plan is not fully implemented. |
| [`lookup_table_networks.md`](./lookup_table_networks.md) | Implemented | Key commits: [`904d10f`](https://github.com/mensch72/empo/commit/904d10f), [`70ee712`](https://github.com/mensch72/empo/commit/70ee712) | Lookup-table Phase 2 networks are present under `src/empo/learning_based/phase2/lookup/`, even though the plan document still says "Planning". |
| [`mpi_issue.md`](./mpi_issue.md) | Deferred | Tracking issue: [#17](https://github.com/mensch72/empo/issues/17) | The MPI/distributed execution proposal is tracked as an open GitHub issue and has not landed in `src/empo/`. |
| [`network_initialization_strategies.md`](./network_initialization_strategies.md) | Deferred | — | The proposed initialization options are not present in the current Phase 2 config or network factory code. |
| [`parameterized_goal_sampler.md`](./parameterized_goal_sampler.md) | Deferred | — | The codebase still uses the existing goal-sampler abstractions rather than the planned parameterized/Bayesian sampler. |
| [`ppo_a3c_considerations.md`](./ppo_a3c_considerations.md) | Abandoned | Superseded by landed PR: [#136](https://github.com/mensch72/empo/pull/136) | The repository adopted a PufferLib PPO path instead of pursuing PPO/A3C changes directly inside the original Phase 2 trainer. |
| [`pufferlib_ppo_port.md`](./pufferlib_ppo_port.md) | Implemented | Landed PRs: [#136](https://github.com/mensch72/empo/pull/136), [#142](https://github.com/mensch72/empo/pull/142) | The Phase 2 PPO port exists under `src/empo/learning_based/phase2_ppo/` and its follow-up fixes have landed. |
