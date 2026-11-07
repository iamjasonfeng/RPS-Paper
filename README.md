# RPS Paper

Regressive Plasticity Schedule: A Two-Stage Post-Training Schedule for ARC Program Synthesis

[PDF](https://iamjasonfeng.github.io/RPS-Paper/rps_arc_prize_2026_paper.pdf)

Abstract

This paper reports an experimental study of Regressive Plasticity Schedule (RPS), a two-stage post-training
schedule: first train on easier data at a higher learning rate, then train on harder data at a substantially lower
learning rate. Earlier work has studied easy-to-hard curricula combined with learning-rate reductions; the
contribution here is an evaluation of this scheduling principle in LLM post-training for ARC program
synthesis, not the introduction of the general principle. I tested RPS on Qwen3-8B using Alibaba Model
Studio managed DPO fine-tuning with LoRA. The control condition, Equal Plasticity Schedule (EPS), used
the same model, two-stage datasets, within-stage cosine scheduler, and Stage 1 checkpoint, but did not
reduce the Stage 2 base learning rate. On ARC-AGI-1 public evaluation, exact test-output accuracy was
17/419 for RPS versus 10/419 for EPS. On ARC-AGI-2 public evaluation, neither condition solved any test
outputs, but 234/240 attempted programs executed without error for RPS, compared with 188/240 for EPS.
These preliminary results support further investigation of lower Stage 2 learning rates under this curriculum.
They do not establish broad reasoning improvements or isolate the interaction between curriculum ordering
and learning-rate reduction.

Update notice (November 6, 2026): This revision adds discussion and citations of prior work that combined
easy-to-hard curricula with learning-rate reductions. I was unaware of these papers when I independently
developed RPS and conducted the original experiments. The paper now distinguishes this prior scheduling
principle from the experimental study of RPS in LLM post-training reported here. This revision also adds a
citation for Codex's assistance with writing this paper. This paper was originally published on June 5, 2026.
