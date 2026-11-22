# Speaker script, Dec 3 (≈ 10 min)

One sentence per slide: the line that must land, in words a classmate who hasn't read the paper can follow.
Slide text and figures: docs/slides_outline.md (12A/13A are the primary versions). Numbers are means over
seeds; details in docs/results_seeds.md.

| # | slide | the sentence that must land | time |
|---|---|---|---|
| 1 | Chasing the Edge | "When a model trained with RL gets to choose how hard its training problems are, what should it aim for, and does it matter?" | 0:20 |
| 2 | The paper we build on | "A small model learns genuinely new problems from RL only when it trains on problems it sometimes solves, and the paper picks that band, 11 to 14 operations, once for the whole run." | 0:40 |
| 3 | Why the edge matters | "The model answers every problem six times and learns from the difference between right and wrong answers, so if all six agree it learns nothing, and what it can learn from depends on where it is right now." | 0:45 |
| 4 | The edge moves | "Training on 11 to 14 makes 15 to 17 learnable, but the fixed band never goes there." | 0:40 |
| 5 | Three ways to chase the edge | "We built three samplers that aim at three different things: the most usable signal, the fastest improvement, and the most room to improve." | 0:50 |
| 6 | Baseline reproduces | "Our version of the paper's method matches their released model within a point everywhere, so every comparison is against a faithful baseline." | 0:25 |
| 7 | Where samplers went | "τ=0.5 never moves, the bandit gets stuck at 12 to 14, τ=0.3 climbs through the paper's band and stops at its top, and τ=0.1 climbs to 17; each stops at the first level it can't push past its move-up point, so τ=0.1 adapts for the first half of the run and then stays put." | 0:35 |
| 8 | Signal ≠ learning | "Forty percent of problems teaching something, zero new problems; twenty-five percent, ninety-five new problems." *(say the numbers, then pause)* | 0:30 |
| 9 | Where learning is fastest | "A level improves fastest when the model solves it ten to twenty percent of the time and stops at about half, so τ=0.1 climbs through that fast stretch while it lasts and τ=0.5 trains where nothing is left, which corrects our own proposal." | 0:55 |
| 10 | The bandit | "Rewarding improvement fails because the level you train improves *because* you train it, so the bandit keeps rewarding its own choice and never moves on." | 0:50 |
| 11 | Room to improve wins | "On the three hardest levels, which τ=0.1 never trained on, it solves nearly 28 of 102 new problems; uniform sampling, which did train there, solves 20." | 0:50 |
| 12A | Range wins; the climb buys speed | "At a quarter of the compute τ=0.1 already matches the paper's final result, 54 against 48, while the control is at 33 because it spends its first steps on levels nobody can learn yet. The control catches up by step 50, and by the end the range τ=0.1 found, 86 of its 95, is most of the win. The climb buys speed, not final quality." | 1:20 |
| 13A | Cost and next steps | "τ=0.1 gives up about 1.5 points of reliability on easy problems for the most new ability; if you already know the range, sample it uniformly, and if you don't, τ=0.1 finds it." | 0:50 |

Total ≈ 10:00. If short on time, cut slide 6 to one sentence on slide 5 and drop slide 10's speaker note.

## Likely questions

**"Doesn't the 11–17 control show your adaptive method doesn't matter?"**
It shows the *range* matters most, and τ=0.1 is how we got the range. Nobody would have picked 11–17 in
advance: three of those levels are ones the starting model never solves, and the paper's own choice (11–14)
gets 48 against 86. The climb adds a further +8.7 hard problems (better on 17 problems, worse on 5,
p = 0.017), but that is the smaller part, and we say so. And τ=0.1 does not track the edge all the way: it
climbs for the first 45–54 steps and then stays at 17, the first level it can't push past 0.2.

**"Does τ=0.1 really follow the edge?"**
For the first half of the run. It climbs while the level it trains can pass its move-up point of 0.2, and
stops at the first level that can't: level 17, for the last 46–55% of the run, in every seed. The same rule
explains where τ=0.3 (14) and τ=0.5 (11) stop. Its value is the range it finds on the way up. Whether 17 is a
real ceiling or just slow is what the 200-step runs will tell us.

**"Three seeds: is τ=0.1 vs uniform real, or noise?"**
It holds every way we checked. Compared problem by problem over 3 seeds each, τ=0.1 solves 15 problems more
often and uniform 2 (p = 0.002, +7.7 per seed, 95% CI +3.7 to +12.3). The gap goes the same way in each seed
separately (9–1, 8–1, 9–1), and τ=0.1's seeds give 27, 28 and 28. The weaker claim, climb vs range on levels
18–20, is not significant (p = 0.11), and we don't lead with it.

**"Does climbing make the model forget easy problems?"**
It makes it slightly less reliable, not less able. The share of single tries that are right on easy problems
drops 1.5 points, in every seed, and ends 1.6 below the starting model; medium drops 2.5 points. The share of
problems solved at least once in 128 tries is unchanged, and it loses as few problems the starting model could
solve as the other arms (0–1 easy and 1–3 medium per seed). The control, which keeps 57% of its training on
11–14, has no such cost; testing a mix of the two is January work.

*(Backup if asked: "Why not just use τ=0.5, most problems teach something there?" Slide 9: a level's success
rate tops out near 0.5 here, so τ=0.5 gets the most signal where nothing is left to learn. "Only 100
steps?" Half the paper's run; 200-step runs are the first January item that could change the picture.
"What is GRPO?" Group Relative Policy Optimisation: sample several answers to the same problem, reward the
correct ones, and push the model toward answers that beat their group's average.)*
