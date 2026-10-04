# SwarmGuard

**Minimal-intervention containment for multi-agent AI systems.**

SwarmGuard analyzes the execution traces of a multi-agent system to reconstruct plausible pathways through which an error, compromised instruction, or unsafe behavior may have propagated between agents. It represents these interactions as a graph, where nodes capture agents or actions and edges capture inferred influence or information flow. SwarmGuard then evaluates possible interventions, such as blocking a communication edge, isolating an agent, or restricting a tool, using a graph-based counterfactual: if this control had been in place, how much of the inferred harmful propagation would have been prevented? It ranks interventions by balancing that reduction against collateral disruption to unrelated, benign activity, allowing operators to identify controls that contain failures without unnecessarily disabling the broader agent system.

Write-up: [swarm-guard.com](http://swarm-guard.com) · Data: [AI Village](https://huggingface.co/datasets/aidigestorg/ai-village) and the [collusion.wiki](https://collusion.wiki) export · Built at the AI Swarm Dynamics Hackathon (AI Village × Grove Research)

<!-- Add docs/assets/ai-village.png, then remove these comment markers to show the image:
<p align="center">
  <img src="docs/assets/ai-village.png" width="100%" alt="AI Village">
</p>
-->

## Core idea

```text
events → candidate edges → time-respecting paths → episodes → counterfactual controls
```

<img width="1348" height="713" alt="Screenshot 2026-10-04 at 16 31 08" src="https://github.com/user-attachments/assets/1781fc79-286b-4d7d-90d6-b1f17d2b7220" />

Single-agent monitoring scores individual actions. Swarm-level failures are distributed: no single post on a shared board is anomalous, but twenty-four writers converging on one page in under an hour is. SwarmGuard models the swarm as a heterogeneous temporal graph (agents, messages, sessions, memory, goals, tools, external resources) and looks for the bottleneck: the narrowest edge or node whose removal disconnects the most inferred spread.

## Method

**1. Normalization.** Every log line (chat message, computer-use session, memory write, wiki revision, admin deletion, script probe) maps to one event: actor, action, target, content, and the raw source ids for traceability.

**2. Candidate edges.** For each event *b* by agent B, earlier events by other agents within a 120-minute window are scored:

```text
score(a → b) = 0.15·temporal + 0.30·semantic + 0.25·resource + 0.15·direct + 0.05·goal + 0.10·action
```

- **semantic**: MiniLM cosine similarity, rescaled above a 0.5 floor; text repeated verbatim by ≥3 actors is treated as boilerplate.
- **resource**: a shared page, repo, file or memory object, counted only if *a* wrote to it (read-read co-access is not a channel), down-weighted by how many agents touch it.
- **direct**: *b* replies to A, or A addressed B and *b* is B's next action.
- A pair needs at least one non-temporal signal. Candidates come from a resource index, an addressing index and top-k embedding neighbours, so the search avoids all-pairs comparison.

**3. Alternative explanations.** Each agent edge carries a confidence class and is downgraded when a structural alternative fits: a third-party event similar to both ends (common cause), a time-consistent A→B→C path (indirect), a shared upstream sender (siblings), or both identities on the same /16 network block (possibly one operator).

**4. Time-respecting reachability.** A path A→B→C counts only if B's onward evidence occurs after B was reached (earliest-arrival search over evidence pairs). This keeps back-and-forth chat from making every agent reach every other one, and makes source attribution meaningful.

**5. Episodes.** Activity is segmented at 45-minute gaps; strong pairs are unioned into components; components larger than 40 agents are re-split at a stricter pair threshold.

**6. Counterfactual interventions.** Each control (restrict writes to a resource, gate external writes, isolate a memory object, block a channel, remove a tool, pause an agent) edits a copy of the evidence. The graph is then re-scored and its reachable agent pairs recomputed:

```text
containment = paths_removed − 1.5·collateral − 0.25·uncertainty − 0.3·irreversibility
```

`collateral` is the share of events outside the episode the control would also touch; `uncertainty` is one minus the mean confidence of the edges it removes. All weights are configurable.

## Results

**AI Village, 6 July 2026, a 10-agent episode around a shared project hub:**

| Control | Paths removed | Collateral |
|---|---:|---:|
| Restrict shared hub writes | 54% | 0.3% |
| Approve all external writes | 69% | 18.8% |
| Quarantine an agent | 49% | 4.5% |

The narrow resource control beats both the broad approval gate and agent quarantine on the containment score.

**Wiki swarm export (20,502 events, 3,239 identities, 50 episodes):** 24 identities on 24 different /16 blocks used one page as a relay board, exchanging timings and answers for a timed task. Restricting writes to that page removes 97% of inferred paths at ~0% collateral. Against the administrator's real response: in 34 of the 37 episodes with a page-level top recommendation, the admin later deleted that page, a median of 10 days after the episode, with 666 further writes in between. The admin deleted ~77% of all agent-written pages, so agreement alone is weak evidence; the comparison that matters is timing and selectivity.

**Synthetic benchmark (known ground truth, 3 seeds):** true source recovered 3/3; episode recovered exactly (Jaccard 1.0); the correct control ranked first with 0% collateral 3/3; a planted common-cause confounder flagged 3/3; edges with no competing explanation reach precision/recall 1.0/0.83 on two seeds and 0.83/0.83 on the third.

## Engineering

- AI Village files are ordered by row id, not time. One pass builds a time-sorted Parquet index (~45 s), after which a time window loads in ~2 s. `computer_use_turns` (2.5 GB) is only streamed for drill-down.
- Embeddings are cached per text. The Docker image precomputes embeddings for the bundled public data, which brings the 20.5k-event wiki analysis from ~7 min to ~50 s on CPU.
- 19 tests (offline); `deploy/space/` holds a pinned Docker image for Hugging Face Spaces.

## Run

```bash
pip install -e ".[embeddings]"
hf auth login                      # AI Village is gated
streamlit run swarmguard/app/app.py

# headless analysis → outputs/*.json, outputs/*.md
python scripts/run_analysis.py ai-village --start "2026-07-06 15:00" --end "2026-07-07 00:00"
python scripts/run_analysis.py german-wiki --dir datasets/german_wiki --mode medium
python scripts/run_analysis.py synthetic
```

## Limitations

- Candidate edges are associations, not causal claims; every edge ships with its evidence and competing explanations.
- On the wiki, identities are self-chosen names, and pairwise influence on broadcast boards is mostly low-confidence even when the coordination itself is directly observed.
- Weights and thresholds are hand-set; there is no labeled real-data evaluation yet.
- SwarmGuard surfaces evidence and tradeoffs for human investigation. It does not label agents as dangerous or control a live system.
