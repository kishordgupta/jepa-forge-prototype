# Repository privacy audit - September 16, 2026

**Publication status, September 16, 2026:** with the owner's explicit approval, `main` was replaced with the tested, sanitized `automatic-task-selection` branch at commit `b718d2ac68cdc722a38b8a284da675f742ea6b15`. The extension is published on both branches. The metadata-bearing commit is no longer in either branch's ancestry. This removes its reachability through these branches; it does not guarantee deletion of GitHub-retained objects or earlier copies.

Scope: `kishordgupta/jepa-forge-prototype`, its published branch ancestry, the corresponding local source history, and the new publication snapshot. This is not an audit of every repository or service in the owner's account.

## September 17 experiment and manuscript extension

Before publication, the new extension, official-model transfer results, manuscript PDF, LaTeX archive, and anonymous evidence archive passed the working-tree guard: **150 publication candidates, 298 extracted text records, zero credential or private-metadata findings**. Archive manifests were verified, and a fresh extraction regenerated all 12 new evidence outputs byte for byte. The expanded local test suite passed all 173 tests. Separate experiment checks replayed 1,392 CPU/MPS checkpoint snapshots and rescored 1,044 saved test prediction files.

The existing reachable history was also rechecked: 116 unique blobs and 138 extracted records passed, and Gitleaks 8.30.1 found no leaks in the four reachable commits. The staged-snapshot hook and post-commit history scan are part of this publication procedure. Raw downloaded archives, model checkpoints, local cache links, private project URLs, recovery files and host-identifying metadata are excluded from publication. The repository keeps its existing visibility and sanitized ancestry; this update is an ordinary fast-forward commit.

These are bounded scan and integrity results. They do not guarantee the absence of every possible sensitive fact, erase retained historical objects, or replace the existing safeguards for future uploads. GitHub Actions repeats the checks on the published revision; its status must be assessed separately from the local results.

## Findings

* **No credentials detected** by Gitleaks 8.30.1 in the two original local commits, or by Gitleaks plus metadata checks across 82 unique historical blobs (91 extracted text records). This is a bounded scan result, not proof that arbitrary sensitive content can never be present.
* **Two reports contained personal machine metadata:** `results/test_report.xml` and `results/expanded_test_report.xml` had a `hostname` attribute. The value is deliberately not reproduced here. No API key was involved in this finding.
* The expanded-results gzip, report PDFs (text and document metadata), and the historical source ZIP were inspected. The reviewed publication contained no matched private Overleaf project link, personal home-directory path, personal email, or private-key material.
* Before cleanup, GitHub showed one branch, no tags, no releases, and no forks. Repository visibility is **private**. The published history before this cleanup consisted of an initialization commit and the prototype publication commit. The earlier publication receipt records an exact match between the local and remote source trees. The extension subsequently added the `automatic-task-selection` branch; both branches now use sanitized ancestry.

## Remediation

The two XML files have been anonymized; experiment scores and test outcomes are unchanged. Their affected publication checksum was regenerated. A clean historical snapshot is preserved as commit `986d88b6e217ab281263bcac8808532222a27718`, with tree `a41c757dbf4224416e0d190779965a3767202098`, based on the original clean initialization commit. Both published branches use this sanitized ancestry. After explicit owner approval, main was moved from the metadata-bearing commit `f8c5c421f23066641f7a3a3aae838263f1ed8231` to `b718d2ac68cdc722a38b8a284da675f742ea6b15`. GitHub's main history was then verified as the clean initialization, sanitized historical snapshot, extension implementation, and documentation update. The old metadata-bearing commit is absent from that ancestry. Subsequent report updates are ordinary fast-forward commits on the sanitized lineage.

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

## CI verification

Before main was replaced, [GitHub Actions run 35142228793](https://github.com/kishordgupta/jepa-forge-prototype/actions/runs/35142228793) tested extension commit `785fdb2a48dee0798a0413db4fab160e1ac676e6`: **149 tests passed, 20 cache-dependent public-data integration checks skipped**. The staged snapshot passed with 94 files and zero findings, while the all-branch history check correctly flagged the two XML reports on the old main. No allowlist or suppression was added. This historical failure records the pre-cleanup state.

After the approved replacement, [main-branch Actions run 35179069548](https://github.com/kishordgupta/jepa-forge-prototype/actions/runs/35179069548) passed both jobs at commit `b718d2ac68cdc722a38b8a284da675f742ea6b15`. The test job reported **149 passed, 20 skipped**. Privacy checks passed for the current 94-file snapshot (116 extracted text records) and all 111 unique historical blobs (133 extracted text records), with zero credential or metadata findings. Gitleaks separately scanned all four reachable commits and reported no leaks. The 20 skips are public-data cache-dependent integration checks; the corresponding local suite passed all 169 tests. The report-only update following this run leaves the validated experiment code and results unchanged.
