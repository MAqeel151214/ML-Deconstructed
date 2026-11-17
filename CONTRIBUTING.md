# Contributing to ML Deconstructed

Thank you for helping make machine learning easier to understand. ML
Deconstructed is an open, dependency-ordered curriculum that connects
intuition, visible data, mathematics, hand calculation, and executable NumPy
code.

Contributions are welcome from developers, engineers, educators, technical
writers, learners, and anyone with a useful suggestion or new idea. You can
propose a lesson, improve an explanation, correct a mathematical detail, add a
better visual, improve accessibility, or report a reproducibility problem.

## Before you start

1. Search existing issues and lessons so that your idea fits the curriculum
   and is not already in progress.
2. For a substantial change, open an issue first with the lesson or area,
   learner problem, proposed approach, and any relevant references.
3. Keep changes focused. A lesson should teach one clear concept and preserve
   the project's dependency order.

## Local setup

The supported development environment is Python 3.11 with the pinned
dependencies in `requirements.txt`:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a lesson interactively with Jupyter, or run the same checks used by CI
from the repository root:

```sh
pytest --nbmake modules/
find modules -type f -name 'verify_hand_calc.py' -print0 | \
  xargs -0 -r -n1 python
```

If a command reports that there are no matching notebooks or verification
scripts, that is the expected no-content case used by CI.

## Adding or updating a lesson

Published lessons live under `modules/` and normally contain:

- `notebook.ipynb`;
- a purposeful image or other lesson asset when a visual is useful; and
- `verify_hand_calc.py` for deterministic teaching calculations.

Use the existing lessons and `README.md` as the public conventions. A lesson
should state its prerequisites, learning outcomes, assumptions, and estimated
time; introduce notation before using it; show a complete small example; work
through important arithmetic; use clear NumPy-first teaching code where
appropriate; include executable assertions; explain failure modes; provide
exercises; and cite real, stable references.

Do not add a lesson to `_toc.yml` or call it published until its notebook and
verification checks pass and the lesson has received a content and learner
review. Keep notebook outputs reproducible and avoid committing `_build/`,
cache directories, private notes, prompts, raw model output, or secrets.

## Pull requests

Please include:

- a short description of the learner-facing problem and the change;
- the lesson/module affected;
- verification commands run and their result; and
- references for mathematical, implementation, or factual changes.

Keep prose, examples, diagrams, and code original. Do not copy source
material. Review equations, arithmetic, links, captions, colour choices, and
alt text before requesting review. Pull requests should pass the repository's
CI checks, including notebook execution and hand-calculation verification.

## Ideas, questions, and bug reports

Suggestions are useful even when they are not fully formed. Please explain
what would help a learner, contributor, or reader, and include a minimal
reproduction for execution or rendering bugs. Small improvements and
constructive review are just as welcome as complete new lessons.

By contributing, you agree that your work is provided under the repository's
[MIT License](LICENSE).
