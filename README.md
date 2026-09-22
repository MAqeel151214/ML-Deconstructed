# 🧠 ML Deconstructed

> Machine learning taught as a chain you can inspect: intuition → visible data → derivation → hand calculation → executable code.

[![CI](https://github.com/MAqeel151214/ML-Deconstructed/actions/workflows/ci.yml/badge.svg)](https://github.com/MAqeel151214/ML-Deconstructed/actions/workflows/ci.yml)

📖 **Read online:** <https://MAqeel151214.github.io/ML-Deconstructed/>

## What makes this different

Each published lesson uses a small, fully visible example to connect the mathematics to code. The core teaching standard is:

1. explain the problem in plain language;
2. define notation before using it;
3. derive the important formulae step by step;
4. work through a complete hand calculation;
5. implement the idea in clear NumPy code;
6. verify deterministic teaching calculations with executable assertions; and
7. show pitfalls, exercises, references, and a useful visual explanation.

Where a meaningful equivalent exists, lessons also compare the teaching implementation with an established library. Animations are used when motion is essential to the concept; a clear static visual is used otherwise.

## Who this is for

Start with Module 00 if you know basic Python but want to build the mathematical intuition behind machine learning. You should be comfortable with variables, lists, functions, and elementary arithmetic. Each lesson states its specific prerequisites and learning outcomes.

## Curriculum

| Module | Lessons |
|---|---:|
| 00 — Math Foundations | 8 |
| 01 — ML Fundamentals | 6 |
| 02 — Supervised Regression | 8 |
| 03 — Supervised Classification | 10 |
| 04 — Ensemble Methods | 8 |
| 05 — Unsupervised Learning | 8 |
| 06 — Neural Networks & Deep Learning | 10 |
| 07 — Model Evaluation & Selection | 7 |
| 08 — Feature Engineering & Preprocessing | 7 |
| 09 — Capstones & Case Studies | 5 |

Published lessons appear in the online table of contents as they pass the project’s verification and review checks.

## Run a lesson locally

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run a lesson notebook in Jupyter, or use any browser-launch link included in that lesson. Browser launch links are provided only for platforms that are actively maintained for the lesson.

## License

This repository is released under the [MIT License](LICENSE).
