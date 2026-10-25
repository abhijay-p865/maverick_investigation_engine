# MaverickX Working Rules

These rules apply to every contributor and automated agent working in this repository.

1. Read `PRD.md` before making product or scientific decisions. It is the authoritative specification.
2. Explain completed work in plain language: what was built, what it does, and why it matters.
3. Work in manageable, runnable chunks. Keep the repository usable at the end of each chunk.
4. Take ownership of technical execution: write code, install available dependencies, inspect data, run experiments, and fix errors wherever possible.
5. Test actual behavior. Never report that something works without running the relevant checks.
6. Ask the project owner only when access, installation approval, or an important product decision is genuinely required.
7. Never fabricate data, simulator output, experimental results, or model performance. Clearly label examples and placeholders.
8. Keep `PROGRESS.md` current with completed tasks, decisions, test results, blockers, and next steps.
9. Keep the model independent from presentation code. A lightweight website is now in scope for CSV upload and report display, but model logic must remain usable from Python and the command line.
10. Never claim that this prototype detects microbial contamination. Initial benchmark faults are simulated process and sensor faults.
11. Keep ground-truth fault labels, fault intervals, and batch outcomes out of model features. Use them only for evaluation.
12. Split data by complete batch, never by individual time-series rows.

For each development chunk, communicate:

- At the start: **Goal**, **Why**, and **Expected output**.
- At the end: **Completed**, **Verified**, **Issues**, and **Next**.

