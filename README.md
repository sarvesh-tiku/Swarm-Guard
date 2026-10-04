
'<img width="1333" height="705" alt="Screenshot 2026-10-04 at 16 35 42" src="https://github.com/user-attachments/assets/af32fdd5-a103-4927-9e29-9ab0a30a35f7" />
# SwarmGuard

**Minimal-intervention containment for multi-agent AI systems.**

SwarmGuard reconstructs how behavior propagates across agents and shared resources, then identifies the smallest reversible control that can interrupt that propagation.

<!-- Add docs/assets/ai-village.png, then remove these comment markers to show the image:
<p align="center">
  <img src="docs/assets/ai-village.png" width="100%" alt="AI Village">
</p>
-->

Built on the [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village).

## Core idea

```text
agents → resources → propagation graph → intervention
```

<img width="1348" height="713" alt="Screenshot 2026-10-04 at 16 31 08" src="https://github.com/user-attachments/assets/1781fc79-286b-4d7d-90d6-b1f17d2b7220" />


Single-agent monitoring asks: *Is this agent doing something unsafe?*

SwarmGuard asks: *Where is the propagation bottleneck?*

It infers time-respecting pathways across messages, tools, goals, memory, and external resources, then tests counterfactual controls such as:

- restricting a shared resource
- blocking a communication edge
- limiting a tool
- isolating memory
- quarantining an agent

Each intervention is ranked by containment, collateral impact, uncertainty, and reversibility.

## Example

In one 10-agent AI Village episode:

| Control | Paths removed | Collateral |
|---|---:|---:|
| Restrict shared hub writes | 54% | 0.3% |
| Approve all external writes | 69% | 18.8% |
| Quarantine an agent | 49% | 4.5% |

The point is not to shut down the most visible agent. It is to find the smallest cut that stops the spread.

## Run

```bash
pip install -e ".[embeddings]"
hf auth login
streamlit run swarmguard/app/app.py
```

## Data

[AI Village on Hugging Face](https://huggingface.co/datasets/aidigestorg/ai-village)

SwarmGuard does not claim causality or label agents as dangerous. It surfaces evidence, alternative explanations, and intervention tradeoffs for human investigation.
