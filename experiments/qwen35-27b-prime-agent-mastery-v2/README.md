# Qwen3.5 27B Prime Agent Mastery Battery V2

This is the official-Prime-Agent successor to the historical 74-task v1 battery.
It keeps the same six slice sizes (10 + 20 + 20 + 8 + 8 + 8) and the same
coordination, ownership, and Oolong task meanings. The foundation slice is versioned
separately because the current Prime Agent ACP runtime removed the private metadata
used by v1. It pins the published beta whose suffix identifies upstream Prime Agent
commit `97b994c`; the previously anticipated `0.7.3` artifact was never published.

The v2 foundation tasks score only observable behavior: exact IPython execution,
kernel persistence, ACP conversation resume, explicit child-result delivery, and
child cancellation with runtime side-effect verification. The battery does not use
the retired `thinking`, `autonomous`, or `gates` harness options. Thinking is requested
through the standard sampling `reasoning_effort` field.

Do not edit these files while comparing model sizes or checkpoints. Create a new
versioned battery instead. V1 and v2 scores are not directly interchangeable; v1 is
the historical Prime Agent 0.7.1 boundary and v2 is the frozen current-ACP boundary.

V2 is a frozen regression comparator, not the complete Stage-0 teacher-promotion
gate. Its matched ownership arms test useful versus unnecessary delegation, and its
follow-up and handshake arms test causal bidirectional control. Malformed child
results, delayed replies, and recoverable messaging faults require a separate,
versioned resilience suite. A checkpoint must pass both the frozen comparator and
that resilience suite before it can be called the canonical teacher.
