# Draft email to the paper's authors (for Eren to send; not sent)

**To:** the corresponding author(s) of arXiv 2512.07783 (Zhang, Neubig, Yue)
**Subject:** Question about the released `extrapolation_rl/.../rl/op11-14_uniform` checkpoint (reward config, steps)

---

Dear Dr. Zhang, Prof. Neubig and Prof. Yue,

We are two students at Koç University building a course project (COMP550) on top of
"On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models".
Thank you for releasing the code, data and checkpoints. They made it possible for us to
reproduce the base-model numbers (Fig. 3 / Fig. 9) for
`id2-10_0.2easy_0.3medium_0.5hard/base` within a few points on our cluster.

We are now reproducing the op 11–14 RL run as our baseline. We would like to compare it
against `Interplay-LM-Reasoning/extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/rl/op11-14_uniform`,
and we could not tell from the repository which configuration produced that checkpoint:

1. **Reward.** The config whose name matches the checkpoint,
   `scripts/composition/op-difficulty-10B/rl-200steps/op11-14_uniform.yaml`, uses the
   answer-only reward (`verl/dataset.py::compute_score`). The strict process-verified reward
   (`verl/reward_fn.py::compute_score_with_step_process`, `zero_on_process_mismatch: true`) is in
   `op11-14_uniform_process_strict.yaml`. Section 6 reads as if the Section 3 / Fig. 3 runs used the
   outcome-only reward, with process supervision added as a variant there. Was the released
   checkpoint trained with the answer-only reward, the strict reward, or something else?

2. **Training length.** Appendix A.3 mentions two training epochs. The config sets `total_epochs: 1`
   over 4 × 50k heldout problems at batch size 1024 (about 195 steps, in a folder named `rl-200steps`),
   and its `experiment_name` is `rl-op11-14_uniform-100steps`. How many steps was the released
   checkpoint (and the Fig. 3 "RL op=11-14" curve) trained for?

3. **Starting point.** That yaml's `model.path` points at `.../id2-10_uniform`, and its
   `default_local_dir` points at the `0.4995easy_0.4995medium_0.001hard` mixture. We assume both are
   template leftovers and that the released checkpoint starts from
   `id2-10_0.2easy_0.3medium_0.5hard/base` (the `run_r11-14_process_strict.sh` script uses `pt/checkpoint-22157`).
   Is that right?

4. **Evaluation of Fig. 3.** Are the Fig. 3 pass@k values computed on the `test/` split with
   T=0.7, 1024 new tokens and 128 samples per problem, scored with `scripts/score_dependency_graphs.py`?

Short answers would help us a lot. We will cite the paper and the release, and we are happy
to share our results when the project is done.

Best regards,
Eren Acar Basaran and Turan Berke Yakarer
Koç University, COMP550

---

*Note for Eren:* the evidence for each question is in `docs/gate3.md` (section "Released checkpoint:
what we know"). Our released-checkpoint numbers under both scorings are in
`results/gate3_released_rl_comparison.csv`. Mention them in the email if you want to show what we see.
