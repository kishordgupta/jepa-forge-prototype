# Repository privacy audit - September 16, 2026

Scope: `kishordgupta/jepa-forge-prototype`, its published branch ancestry, the corresponding local source history, and the new publication snapshot. This is not an audit of every repository or service in the owner's account.

## Findings

* **No credentials detected** by Gitleaks 8.30.1 in the two original local commits, or by Gitleaks plus metadata checks across 82 unique historical blobs (91 extracted text records). This is a bounded scan result, not proof that arbitrary sensitive content can never be present.
* **Two reports contained personal machine metadata:** `results/test_report.xml` and `results/expanded_test_report.xml` had a `hostname` attribute. The value is deliberately not reproduced here. No API key was involved in this finding.
* The expanded-results gzip, report PDFs (text and document metadata), and the historical source ZIP were inspected. The reviewed publication contained no matched private Overleaf project link, personal home-directory path, personal email, or private-key material.
* GitHub showed one branch, no tags, no releases, and no forks. Repository visibility is **private**. The published history before this cleanup consisted of an initialization commit and the prototype publication commit. The earlier publication receipt records an exact match between the local and remote source trees.

## Remediation

The two XML files have been anonymized; experiment scores and test outcomes are unchanged. Their affected publication checksum was regenerated. A clean historical snapshot is preserved as commit `986d88b6e217ab281263bcac8808532222a27718`, with tree `a41c757dbf4224416e0d190779965a3767202098`, based on the original clean initialization commit. The automatic-selection publication is based on that sanitized ancestry and replaces the affected main-branch history. The old metadata-bearing commit is not an ancestor of the new publication.

The local `main` history has also been rebuilt from the sanitized snapshot. An ignored local recovery bundle and tool-owned recovery references remain local; they are excluded from publication and from the definition of publishable branches/tags. No real credential was found, so no account key was revoked or rotated.

The old commit may remain retained by GitHub and accessible by its object ID even after the branch is rewritten. Previously downloaded copies cannot be recalled. Therefore this audit **does not claim that the machine name was never uploaded or has been erased from every retained copy**. If complete platform-side removal of that historical device metadata is required, the repository owner must follow GitHub's [sensitive-data removal procedure](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) and any applicable support process. No support request or message was sent on the owner's behalf.

## Prevention added

* A broader `.gitignore` excludes credential files, private keys, private inputs, caches, data arrays, and checkpoints. Forced additions are still inspected by the staged-snapshot guard.
* A checksum-pinned Gitleaks installer and `scripts/check_privacy.py` scan staged snapshots, working-tree publication candidates, and all unique blobs reachable from branches, tags, and remote-tracking references.
* ZIP/gzip members and PDF text/metadata are extracted for scanning; PNG metadata are checked. Unsupported binaries and incomplete extraction fail closed. Matched values are never printed in the privacy report.
* Commit and push hooks are enabled in this checkout. A newly cloned checkout must install the hook configuration following `SECURITY.md`.
* GitHub Actions scans the checked-out snapshot and full fetched history, then runs credential-history scanning and the test suite. Workflow actions are pinned to reviewed commit IDs.
* A generated fake credential was blocked by the actual Gitleaks engine, and the output did not expose its value. Unit tests cover private links, personal paths/emails, nested archives, ignored credential filenames, and missing scanner behavior.
* `scripts/run_tests_private.py` strips machine and source-file attributes from publishable JUnit XML. The privacy guard checks the output again.

## Platform limits observed directly

GitHub's Advanced Security settings did not offer native secret-scanning or push-protection controls for this private repository. The branch-protection form explicitly states that its rules will not be enforced on this private repository until it is moved to an eligible Team or Enterprise organization account. No account plan was purchased or changed, and no unenforced rule was represented as protection.

Local hooks can be bypassed; API/UI uploads do not invoke them. CI starts after content reaches GitHub. These controls reduce risk but cannot guarantee that every future action is free of sensitive information. The validated publication process must run the local scan before uploading objects. Pixels inside images and arbitrary unstructured meaning are outside the scanner's automated coverage.

GitHub commit authorship continues to show the owner's existing public GitHub identity. This is normal repository attribution, not a credential. Local commits use GitHub's noreply address rather than a personal email.
