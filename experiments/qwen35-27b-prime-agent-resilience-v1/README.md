# Qwen3.5 27B Prime Agent Resilience V1

This frozen 12-task suite complements, but does not modify, the mastery V2
comparator. Its calibration and held-out slices test three observable recovery
boundaries: malformed child results, delayed child completion, and message-type
repair after a real error. Both slices pin the same published Prime Agent beta as
mastery V2.

A teacher candidate must produce complete valid summaries for both mastery V2 and
resilience V1. Keep these task definitions unchanged when comparing the untouched
base and trained checkpoints.
