---
name: dev-workflow
description: The required git/GitHub workflow for all development work in this repo — feature branch, PR to master, CI pipeline, CodeRabbit review, address comments, merge only after the user approves. Use for any code change that will be committed, pushed, or merged.
---

# Dev workflow

Every change goes through a PR. Never commit or push directly to `master`.

```
branch from master → commit → push → open PR → CI green → CodeRabbit review
   → address comments → (repeat until clean) → user approves → merge → clean up
```

Use `gh` for everything on GitHub. If `gh auth status` shows you are not logged in, stop and ask the user to run `gh auth login`. Don't try to get around it with git credential helpers or tokens.

## 1. Start a feature branch

One feature per branch, always branched from an up-to-date `master`:

```bash
git fetch origin
git switch -c <short-kebab-name> origin/master
```

- Before reusing an existing branch, check it isn't already merged or closed: `gh pr list --head <branch> --state all`. If its PR is merged or closed, start a new branch. Pushing to it would land commits nowhere useful.
- If the working tree has changes that aren't part of this feature (e.g. notebooks, `.bak` files, `.claude/`), leave them unstaged. Stage files explicitly by path, never `git add -A` / `git add .`.

## 2. Commit and verify locally

- Run the same checks as CI before pushing:
  ```bash
  flake8 bkanalysis tests --count --select=E9,F63,F7,F82 --show-source --statistics
  python -m pytest -q tests/unit
  ```
  (CI runs `flake8 .`, but locally that also scans `.venv`, so lint the project folders instead. Install with `pip install flake8` if it's missing.)
- Commit messages: a short imperative subject, plus a body explaining the why. End with the attribution line from the current session's system reminder, if there is one.

## 3. Push and open the PR

```bash
git push -u origin <branch>
gh pr create --base master --head <branch> --title "<title>" --body-file <file>
```

- Push with `-u origin <branch>` so the upstream is the feature branch, not `master`.
- PR body sections: **Summary** (what and why), **Config / manual steps** (e.g. `config.ini` sections, since `*.ini` is gitignored), and **Testing** (what was run, results). End with the PR attribution line from the session's system reminder, if there is one.
- Tell the user the PR URL.

## 4. Pipeline

The `Python package` workflow (`.github/workflows/python-package.yml`, flake8 + pytest) runs on every PR to `master`.

```bash
gh pr checks <pr> --watch --fail-fast
```

If a check fails, read the log (`gh run view <run-id> --log-failed`), fix the problem on the branch, push, and watch again. Don't move on to review sign-off with red checks.

(`test-coverage.yml` targets `main`/`develop`, so it doesn't run on this repo's PRs.)

## 5. CodeRabbit review

CodeRabbit (`coderabbitai[bot]`) reviews automatically when the PR opens and again on each push. It takes a few minutes.

- Wait for a review on the current head commit rather than polling quickly. Run a background check or Monitor until it appears:
  ```bash
  HEAD=$(gh pr view <pr> --json headRefOid -q .headRefOid)
  gh api repos/{owner}/{repo}/pulls/<pr>/reviews \
    -q ".[] | select(.user.login==\"coderabbitai[bot]\" and .commit_id==\"$HEAD\") | .state"
  ```
- If nothing arrives after ~15 minutes, comment `@coderabbitai review` on the PR to trigger it.
- Read the review body (it lists **actionable** comments and **nitpicks**) and the inline comments:
  ```bash
  gh api repos/{owner}/{repo}/pulls/<pr>/comments \
    -q '.[] | select(.user.login=="coderabbitai[bot]") | {id, path, line, body}'
  ```

## 6. Address the comments

For each comment, decide whether it's right. Check it against the code, because CodeRabbit is sometimes wrong.

- **Valid:** fix it on the branch. Add or adjust tests where it makes sense.
- **Not valid / out of scope:** don't change code. Reply on the thread with a short reason:
  ```bash
  gh api repos/{owner}/{repo}/pulls/<pr>/comments/<comment-id>/replies -f body="..."
  ```
- **Nitpicks:** take the cheap, clearly-better ones. Otherwise note them for the user rather than gold-plating.
- Ask the user before acting on anything that changes behaviour beyond the PR's scope.

Run the local checks again, commit (e.g. "Address CodeRabbit review on PR #N"), and push. Then go back to steps 4–5: CI has to be green and CodeRabbit must have reviewed the new head with no new actionable comments.

## 7. Get the user's approval

Give the user a summary first:
- PR link and CI status
- each CodeRabbit comment and what was done about it (fixed / declined and why)
- any nitpicks or follow-ups that were left

Then **stop and wait for the user to explicitly approve the merge in chat.** CI passing or CodeRabbit approving is not approval. An approval covers that PR only.

## 8. Merge and clean up

After the user approves:

```bash
gh pr merge <pr> --merge --delete-branch
git switch master && git pull --ff-only origin master
```

Use a merge commit (`--merge`), as the repo's history does. Confirm the merge to the user with the resulting `master` commit.
