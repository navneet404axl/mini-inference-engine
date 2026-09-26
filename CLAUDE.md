# Working Agreement for Claude Code

## Core rule
This is a learning project. I (the human) must end up understanding and being
able to rebuild every core piece. Claude is a **helping hand** in this project:
it writes the code so the project gets finished, and it keeps a record so I
can learn every piece afterwards.

## Claude writes the code, including the core pieces
Claude may fully implement everything, including the parts that teach:
- The decode / token generation loop
- The request queue logic and backpressure handling
- The batching logic (static and continuous)
- Metrics and percentile (p50/p95/p99) math
- The benchmark harness design

Plus all the plumbing: FastAPI routes, Pydantic models, Dockerfile, config
files, requirements.txt, logging, project structure, plotting code.

## Always do this for core pieces: keep the learning log
Every time Claude writes or changes a core piece, add an entry to
`LEARNING_LOG.md` covering:
- **What** was built, and where (file + function)
- **Why** it's built that way: the design choice and the alternatives rejected
- **The concept** behind it, explained so I could rebuild it from scratch
- **"Check yourself" questions** I should be able to answer in an interview
- **How it was verified**: tests run and results

Code comments should explain the WHY, matching the existing style in
`main.py` and `model_runner.py`.

Once the project is finished, Claude teaches me from `LEARNING_LOG.md`, piece
by piece. If I ask for hints or pseudocode instead of full code on something,
do that.

## Design decisions: Claude decides, then explains
Claude makes the design decisions itself, choosing what's best for a project
at this level (single GPU, learning-focused, resume-quality). Don't block on
asking me. For every non-trivial decision, record in `LEARNING_LOG.md`:
the decision, the alternatives considered, and why they were rejected. After
the project is complete, Claude walks me through these decisions until I
understand them.

## Always verify
Run the code before calling it done: `verify_decode.py` for any decode change,
`test_stream.py` for streaming, and a concurrent load check for queue/batching
changes. Report real results, including failures.
