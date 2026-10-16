<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explainability Subcard

Field | Response
:---|:---
Intended Task/Domain: | Research
Model Type: | Diffusion Transformer (DiT)
Intended Users: | Researchers and developers studying, evaluating, or extending LiDAR world models and sensor simulation.
Output: | 3D LiDAR range-view video sequences with flexible resolution and length; streaming supported.
Describe how the model works: | A fully attention-based tokenizer maps LiDAR range sequences from different scanning patterns into a shared latent space. A diffusion transformer predicts future latent sequences conditioned on a LiDAR prefix, HD map, persistent spatial cache, and optionally text. A four-step causal student uses temporal K/V and world-aligned spatial caches for real-time, long-horizon autoregressive rollouts.
List the groups (or group characteristics) for which this was tested to produce comparable outcomes: | Not Applicable
Technical Limitations and Mitigation: | Performance may degrade during very long-horizon rollouts, and generated points at very far distances may be noisy.
Verified to have met prescribed NVIDIA quality standards: | Yes
Performance Metrics: | L1<br>Chamfer Distance<br>FUD<br>MMD<br>JSD<br>TTCE<br>CTC<br>ICP-Avg<br>Throughput<br>Latency
Potential Known Risks: | The model may generate inaccurate geometry, depth, or object motion, reducing the fidelity of simulated LiDAR data and potentially leading to incorrect research conclusions if outputs are used without validation.
Licensing: | None
