# Speaker script, Dec 3 (≈ 10 min)

One sentence per slide: the line that must land. Slide text and figures: docs/slides_outline.md (12A/13A are
the primary versions). Numbers are means over seeds; details in docs/results_seeds.md.

| # | slide | the sentence that must land | time |
|---|---|---|---|
| 1 | Chasing the Edge | "When GRPO chooses its own training difficulty, what should it aim for, and does it matter?" | 0:20 |
| 2 | The paper's finding | "RL teaches this model new problems only at its edge of competence, and the paper fixes that edge at op 11–14 for the whole run." | 0:40 |
| 3 | Why the edge matters | "A group whose six rollouts all succeed or all fail contributes exactly zero gradient, so where the model is right now decides what it can learn." | 0:45 |
| 4 | The edge moves | "Training on op 11–14 makes op 15–17 learnable, but the fixed band never goes there." | 0:40 |
| 5 | Three objectives | "We built three samplers that aim at three different things: the most gradient, the fastest improvement, and the most headroom." | 0:50 |
| 6 | Baseline reproduces | "Our fixed band matches the paper's released checkpoint within a point everywhere, so every comparison is against a faithful baseline." | 0:25 |
| 7 | Where samplers went | "τ=0.5 never moves, the bandit stalls at op 12–14, τ=0.3 rediscovers the paper's band, and τ=0.1 climbs to op 17, in every seed." | 0:35 |
| 8 | Gradient ≠ learning | "40% informative gradient, zero new problems; 25% informative gradient, 95 new problems." *(say the numbers, then pause)* | 0:30 |
| 9 | Where learning is fastest | "A level learns fastest around p = 0.1–0.2 and tops out near 0.5, so τ=0.1 trains on the steep part and τ=0.5 sits on the ceiling, which corrects our own proposal." | 0:55 |
| 10 | The bandit | "Rewarding improvement fails because the level you train improves *because* you train it, so the bandit pays itself and never moves on." | 0:50 |
| 11 | Headroom wins | "On the three hardest levels, which τ=0.1 never trained on, it solves nearly 28 of 102 new problems; uniform, which did train there, solves 20." | 0:50 |
| 12A | The range is most of the win | "Sampling τ=0.1's range uniformly gets 81% of its gain: the paper's band gets 48, the range τ=0.1 found gets 86, the schedule takes it to 95. The contribution is finding that range." | 1:10 |
| 13A | Cost and next steps | "τ=0.1 gives up 1.5 points of easy-set reliability for the most new ability; if you already know the range, sample it uniformly, and if you don't, τ=0.1 finds it." | 0:50 |

Total ≈ 9:50. If short on time, cut slide 6 to one sentence on slide 5 and drop slide 10's speaker note.

## Likely questions

**"Doesn't the op 11–17 control show your adaptive method doesn't matter?"**
It shows the *range* matters most, and τ=0.1 is how we got the range. Nobody would have picked op 11–17 in
advance: three of those levels have a base success rate of exactly zero, and the paper's own choice (op 11–14)
gets 48 against 86. The schedule adds a further +8.7 hard problems (17 vs 5 problems, p = 0.017), but that is
the smaller part, and we say so.

**"Three seeds: is τ=0.1 vs uniform real, or noise?"**
It is consistent at every level we checked. Paired by problem across 3 seeds each, τ=0.1 solves 15 problems
more often and uniform 2 (p = 0.002, +7.7 per seed, 95% CI +3.7 to +12.3). The gap goes the same way in each
seed separately (9–1, 8–1, 9–1), and τ=0.1's seeds span 27–28. The weaker claim, schedule vs range on op 18–20,
is not significant (p = 0.11), and we don't lead with it.

**"Does climbing make the model forget easy problems?"**
It makes it slightly less reliable, not less able. Easy pass@1 drops 1.5 points, consistently across seeds and
1.6 below the base model, and medium drops 2.5 points. pass@128 is unchanged, and it loses about as few base-solved
problems as the other arms (0–3 per seed). The control, which keeps 57% of its batch on op 11–14, has no such cost, and
testing a mix of the two is January work.

*(Backup if asked: "Why not just use τ=0.5, it maximises gradient?" Slide 9: at T = 1 a level's success rate
tops out near 0.5, so τ=0.5 maximises gradient where nothing is left to learn. "Only 100 steps?" Half the
paper's run; 200-step runs are the first January item that could change the picture.)*
