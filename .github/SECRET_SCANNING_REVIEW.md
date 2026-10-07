# Reviewed secret-scanning fixtures

The pinned Infisical scanner still runs all built-in rules. The workflow excludes
only a reviewed finding whose exact repository path, rule ID, and SHA-256 digest
of its complete matched text equal an entry in `secret-scanning-exceptions.json`.
The file contains no credential values. No whole files, rules, commits, datasets,
or notebook types are exempt. A changed value or a match in another file still
fails the check. Invalid exception files fail the check as well.

Reviewed on 2026-10-07:

- `datasets/style_mix_untransformed/train.jsonl` (`generic-password`): Historical instruction-style training example contains an illustrative password.

Do not add an exposed credential to this list. Revoke or rotate it and remove it
from source first. Inspect candidate exceptions privately and test that a changed
credential in the same file is detected before accepting an exception.

Raw scanner reports can retain source text despite `--redact`. The workflow reads
and filters them in a temporary directory, publishes only location metadata, and
deletes the report without uploading it. Run **Actions → Secret scanning → Run
workflow** with `full_history` enabled to verify reviewed history. There is no cron
or automatic retry. An unconfigured local CLI scan can still report these fixtures.
