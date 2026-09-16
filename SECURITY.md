# Publication and privacy controls

This repository contains a technical prototype, public/synthetic benchmark evidence, and reports. Private proposal sources, Overleaf project links, account credentials, private keys, personal email addresses, home-directory paths, and machine names must not be committed. Keep private inputs outside this checkout or inside ignored `private/`; keep credentials in environment variables or an appropriate secret store.

Install the local protection after cloning:

```bash
python -m pip install '.[dev]'
python scripts/install_security_tools.py
git config core.hooksPath .githooks
```

The pre-commit hook scans the **staged Git snapshot**, including ignored files forcibly added to Git. The pre-push hook also scans every unique blob reachable through local branches, tags, and remote-tracking references. Both fail when Gitleaks is missing, extraction fails, or a finding is present. They do not upload contents to a scanning service. The installer verifies a pinned release SHA-256 before executing Gitleaks.

Manual checks before any API upload or publication:

```bash
python scripts/check_privacy.py --worktree
python scripts/check_privacy.py
python scripts/check_privacy.py --history
```

The scanner combines Gitleaks' default credential rules with repository-specific metadata rules. It extracts ZIP/gzip content and PDF text/metadata; PNG metadata are checked. Archives are bounded to 64 MiB uncompressed and five nesting levels. Unsupported binary formats and incomplete scans fail closed. Large local checkpoints, arrays, and dataset caches remain ignored. To distribute them, review a separate artifact package; this source-publication scanner deliberately does not approve model/data binaries. Image pixels are not OCR-scanned.

Use `python scripts/run_tests_private.py` for publishable JUnit evidence; it strips device-name and source-file attributes. Running pytest with `--junitxml` directly can reintroduce personal machine metadata. The publication guard catches that metadata before an ordinary commit.

GitHub Actions repeats source and full fetched-history scans and runs the tests. CI runs **after** a push and does not undo exposure from a bypassed local hook. Local hooks can also be bypassed, and API/UI uploads do not invoke them. Repository-native secret scanning/push protection and branch rules depend on GitHub plan availability; their actual observed status is recorded in `SECURITY_AUDIT.md`. No scanner can guarantee that every future manual upload is free of sensitive information.

If a real credential is ever found, stop publication, revoke or rotate it at its provider, remove the material from current files and reachable history, and inspect use of the credential. History rewriting changes commit IDs and does not erase copies, forks, caches, or GitHub-retained objects. Follow [GitHub's sensitive-data removal procedure](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository); platform-retained data may require owner/support follow-up. Never place a discovered secret in an issue, report, or chat transcript.

Report a suspected issue privately to the repository owner using an existing trusted channel; do not open a public issue containing the value. This project does not provide a vulnerability-reporting inbox or promise legal/privacy certification.
