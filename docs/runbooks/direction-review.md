# Direction review runbook (every 6 hours, two reviewers)

User rule (updated by Taeyang 2026-10-08): every 6 hours the current direction is reviewed by TWO independent reviewers, **fable** and **astra**, whatever host runs the session. Both run read-only; neither edits files, launches jobs, pushes or submits.

Before dispatch, re-read Shared state. If another cycle is running or a review already completed for this fixed six-hour scheduled cycle, do not duplicate it. Closely overlapping host completions share the same cycle; they do not postpone the next scheduled cycle by another6hours. Claim `Review dispatch: running — <host> (<slot>, since <time>)` before launching; clear it when recording results. Use `Last research sync` (migrated from `Last 4-hour sync`) and `Last direction review` for deltas. The scheduled C heartbeat is `rsna-6`; job completion monitoring is separate. After review, inspect source-index changes from similar competitions, public notebooks, discussions and papers when useful. Deduplicate discoveries and all plan headings/Hypothesis lines, coordinate with the active collector, and add at most3 evidence-backed hypotheses.

## How to run them

- **fable** (Claude model): from a Claude Code session, launch a background sub-agent with `model: fable`, `subagent_type: general-purpose`, and the prompt below. From a Codex session, use the Claude Code CLI if it is installed (`claude -p --model fable "<prompt>"`); if it is not available, record that in the event.
- **astra** (Codex model `gpt-6-astra`): from any session on this machine,

  ```bash
  codex exec -m gpt-6-astra -s read-only -C "H:\kaggle\competitions\rsna-knee-abnormality-detection" --skip-git-repo-check -o <scratch>/astra-review.md - < <scratch>/review-prompt.txt
  ```

  (`-s read-only` keeps it from writing; `-o` saves its final message.) A Codex session can run it natively.
- Before launching, refresh the facts the prompt needs: a public-leaderboard summary (`kaggle competitions leaderboard rsna-knee-abnormality-detection --download -p .local/lb`; ranks 1/10/20/100/265/531/1000), our submissions and pending scores, and the compute lanes (local GPU queue and running job, Kaggle kernels, Colab).
- Session crons only fire while a session is idle; a busy session must start the review itself at the 6-hour mark (a background timer that ends at that time is a reliable trigger).

## Fallback when a reviewer is unavailable (user rule, 2026-10-07)

- Claude side: `fable` → if it fails (usage limit, error, timeout), retry once with `opus`.
- Codex side: `gpt-6-astra` → if it fails, retry once with `gpt-6-sol` (`codex exec -m gpt-6-sol -s read-only ...`).
- If only one side produces a review, use that one alone. If neither does, log "direction review: both reviewers unavailable (<reasons>)" and continue the work.
- The event names the model that actually reviewed each side and the reason for any fallback.

## Before the reviewers: 6-hour sync (user rule, 2026-10-07 16:28 KST)

Run the sync defined in agents.md ("6-hour sync") first: new events/discoveries from other slots and hosts since `Last research sync`; answers to CHALLENGED/HOLD verdicts on your own discoveries; up to about six cross-checks of other hosts' PENDING discoveries; follow-up plan items from everything new, verified, challenged or corrected; one sync event. Add a five-line summary of the sync (what changed, open challenges, cross-check backlog, new plan items) to the reviewer prompt under "Facts now", so both reviewers judge the direction with the other sessions' latest work in view.

## Prompt (fill the bracketed facts)

> You are an independent, read-only reviewer of the RSNA Knee Abnormality Detection project at H:\kaggle\competitions\rsna-knee-abnormality-detection (Kaggle code competition; 12 binary targets, macro AUC; hidden test ≈1,300 studies, public ≈30% / private ≈70%; 9 h on 2×T4; deadline 2026-10-22 23:59 UTC; 5 submissions/day; the final score is the PRIVATE leaderboard; two final selections). Read agents.md, plan.md, discoveries.md and handoff.md (Shared state, slot sections, the last ~25 events) yourself. Facts now: [leaderboard summary], [our scored and pending submissions], [lane status], [anything new since the last review]. Give an objective diagnosis against the goal (maximise the final private score; a medal, ideally first place) in ≤ 450 words: (1) is the direction right, and the single biggest risk or blind spot; (2) what to stop, start or reorder in the next 24–48 h, with plan item IDs; (3) idle or wasted compute lanes and the best next use of the local GPU and both Kaggle slots; (4) evaluation/overfitting risks and how to pick the two final submissions; (5) one or two higher-ceiling ideas not being pursued, with the evidence for them. Do not edit anything.

## After both answer

Log ONE handoff event with heading `none`: `Action: direction review (Claude Code / fable + Codex / astra): ...`, summarising where the two agree and disagree and what is adopted or rejected and why (provenance rule). Apply plan changes through the normal rules. If a reviewer could not run, say so in the event.
