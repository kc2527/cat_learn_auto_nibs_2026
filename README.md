# cat_learn_auto_2026_s1_lab

Lab (PsychoPy + TMS) repository for the 2026 S1 study.

## Structure
- `code/`: experiment runtime + analysis utilities
- `data/`: de-identified behavioral data only
- `docs/`: lab notes and study docs

## Runtime Input
- Lab runtime prompts for participant ID on each launch.
- Participant ID format is exactly 3 digits (for parity with web).
- Only enrolled IDs are accepted (`002, 077, 134, 189, 213, 268, 303, 358, 482, 527, 594, 639, 662, 707, 729, 875, 943, 998`).
- Condition is auto-assigned by fixed rule in code: first 9 IDs -> `90`, last 9 IDs -> `180`.
