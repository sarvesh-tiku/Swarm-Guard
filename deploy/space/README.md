---
title: SwarmGuard
emoji: 🛡️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Where would you step in? Containment for AI agent swarms
---

# SwarmGuard

Reconstructs how behaviour spreads between AI agents and the pages, sites and memory they share, then points to
the smallest, most reversible place an operator could step in.

- **App:** this Space (tabs: Start here · Summary · Episodes · Investigate · Intervene · How it works · Blog)
- **Write-up:** the Blog tab, or `/app/static/blog.html` on this Space
- **Data:** AI Village (gated; read at runtime with this Space's `HF_TOKEN` secret, never stored in the Space),
  the German wiki export (public, included), and a synthetic test.

The public Space opens on the German wiki export and the synthetic test. AI Village is a gated dataset, so its preset only works where an `HF_TOKEN` with access is set (or when you run the app locally after `hf auth login`).
