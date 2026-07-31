# AAAI-27 Supplement Compliance

## Confirmed rules

The AAAI-27 Supplementary Material Guide permits a separate code/data ZIP and
explicitly recommends a representative subset when the complete dataset does
not fit. It also states that supplementary material is optional, reviewers are
not required to consult it, the paper must remain self-contained, all supplied
material must preserve double-blind anonymity, and submissions must not point
to mutable web-hosted supplementary material.

Official guide:
https://aaai.org/conference/aaai/aaai-27/supplementary-material/

The AAAI-27 Main Technical Track call requires a separate reproducibility
checklist and treats reproducibility, documentation, and code/data sharing as
review considerations.

Official call:
https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/

## Size boundary

The AAAI page does not state a numeric ZIP ceiling. OpenReview's documented
supplementary-upload configuration supports files up to 50 MB, and the active
submission form remains the final authority. This package therefore enforces a
strict compressed size below 50 MiB and should be checked again in OpenReview
before the deadline.

OpenReview documentation:
https://docs.openreview.net/getting-started/hosting-a-venue-on-openreview/enabling-supplementary-material-upload

## Package policy

- No author names, institutions, private hosts, home directories, or secrets.
- No anonymous repository pointer is required to run the sample.
- No full restricted upstream dataset is redistributed.
- The supplied data are development-only representative portions.
- Core benchmark code is materialized as ordinary files; there are no symlinks.
- The main paper must contain every claim essential to acceptance.
