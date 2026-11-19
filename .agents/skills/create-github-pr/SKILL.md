---
name: create-github-pr
description: Analyze branch commits, draft a PR, and open it into master with gh. Also handles post-merge cleanup and end-to-end hotfix/auto-merge (feature → master).
---

# Create GitHub PR from branch commits

Analyze commits and diffs on the current branch, draft a PR title and description, push if needed, and open a pull request into `master` with `gh`. **Cleanup mode** syncs `master` after a merged PR. **Hotfix mode** automates the entire lifecycle (branch → PR to `master` → approve/merge → cleanup).

**Default Behavior:**
When invoked (even with empty trailing text), analyze git state, create/switch to a feature branch if currently on `master`, stage changes (excluding secrets/env files and ephemeral runtime directories), commit, push, and open the pull request into `master` immediately with `gh pr create`.

**Branch strategy:** Direct flow is `feature → master`.

- **`master`** — single long-lived default branch. Open all **feature and fix PRs directly into `master`**.
- Never delete `master`, and never push directly to `master` when opening a PR.

## Mode — read trailing user text

| Trailing text | Mode |
|---|---|
| Empty or `now` / `proceed` / `master` / `to master` | **Create** — commit (if dirty), push, and open PR **into `master`** immediately |
| `hotfix`, `merge`, or `emergency` | **Hotfix (end-to-end)** — commit, push, PR to `master`, immediately approve & merge, and clean up back to `master` |
| `cleanup` or `post-merge` | **Cleanup** — sync `master`, switch to it, delete merged feature branch |
| `draft only`, `preview`, or `dry run` | **Draft only** — show proposed title/body; do not push or create PR |
| `draft pr` or `as draft` | **Create draft PR** — `gh pr create --draft` |
| Contains `base:` or `into:` | Override base branch |
| Contains `title:` | Use as title hint |
| Describes scope | Narrow summary to that scope; still list all commits |

**Default base branch:** `master`.

If trailing text is **cleanup**, skip Create — run **Cleanup mode** only.

### Uncommitted changes (Create into `master`)

When the working tree has changes **outside** ignored local state:

1. Inspect the uncommitted changes and determine an appropriate branch name (`feat/...`, `fix/...`, or `chore/...`).
2. Identify files to stage (strictly exclude `.env*`, credentials, model caches `backend/.cache/`, runtime uploads `backend/uploads/`, and generated vector stores unless explicitly part of sample data).
3. Switch to the feature branch (`git checkout -b <branch>`) if on `master`, stage files, commit, push, and open the PR with `gh pr create --base master`.

---

## Cleanup mode (after PR merged)

**Default:** delete merged local + remote feature branch and sync `master`. Use **keep branch** or **no delete** to skip branch deletion.

### C1 — Pre-check

```bash
git status
git branch --show-current
git remote -v
```

| Situation | Action |
|---|---|
| Uncommitted changes | **Stop** — ask user to commit, stash, or discard |
| Already on `master` | Skip checkout; still fetch + pull; skip branch delete |
| On a feature branch | Remember name for delete |

### C2 — Sync `master`

```bash
git fetch origin
git checkout master
git pull origin master
```

Never force-pull or `reset --hard` unless the user asks.

### C3 — Remove merged feature branch

Confirm merge (`git branch --merged` or `gh pr list --state merged --head <branch>`). Run `git branch -d <branch>`; `git push origin --delete <branch>` if remote remains. Never delete `master`.

### C4 — Report `master` status, previous branch, deletions, and `git log -3`.

---

## Step 0 — `gh` pre-check (Create only)

```bash
command -v gh && gh auth status
```

If `gh` is missing or unauthed: after push, print `https://github.com/<owner>/<repo>/compare/master...<branch>` plus install hint (`brew install gh` → `gh auth login`). Do not fail the flow.

If this workspace is **not a git repo yet**, stop and notify the user — initialize only if requested.

---

## Step 1 — Gather git context

```bash
git status
git branch -vv
git remote -v
git log --oneline -15
git fetch origin master 2>/dev/null || true
git log origin/master..HEAD --oneline
git diff origin/master...HEAD --stat
git diff origin/master...HEAD
```

Analyze **ALL** commits since diverging from `master`.

---

## Step 2 — Draft PR title

- One line, **≤ 72 characters** when possible
- **Imperative mood** — "Add …", "Fix …", "Update …"
- Lead with the user-facing or architectural outcome

---

## Step 3 — Draft PR description

```markdown
## Summary
<2–4 sentences>

## Changes

### Backend (FastAPI / RAG)
- …

### Frontend (Nuxt / Vue)
- …

### Docs / Tooling / Docker
- …

## Commits
| SHA | Message |
|-----|---------|
| `abc1234` | … |

## Test plan
- [ ] …

## Notes / follow-ups
…
```

Do **not** include secrets or `.env` files. Do **not** paste huge diffs.

---

## Step 4 — Branch, commit (if dirty), and push

1. Create feature branch (`feat/` / `fix/` / `chore/`) if currently on `master`.
2. Stage and commit product changes. Never stage `.env*`, credentials, or large binary caches.
3. If `npm run ci` or test scripts are configured, run them and ensure they pass before pushing.
4. `git push -u origin HEAD` — never force-push to `master`.

---

## Step 5 — Create PR

```bash
gh pr create --base master --title "…" --body "$(cat <<'EOF'
## Summary
…
EOF
)"
```

Add `--draft` when requested. If a PR already exists: `gh pr view --json url,title,baseRefName` and report the URL.

After creating the PR with `gh pr create`, **immediately report the PR URL to the user** — do NOT poll or wait in a loop for remote GitHub Actions checks to finish.

---

## Hotfix pipeline mode (`create-github-pr hotfix`)

When invoked with `hotfix`, `merge`, or `emergency`, automate the full lifecycle from branch creation into `master` without pausing for manual approvals:

```
[Local work] → [fix branch] → [PR into master] → [Approve & Merge into master] → [Sync master & delete fix branch]
```

### H1 — Prepare & commit hotfix
1. If on `master` with uncommitted changes:
   - Create fix branch: `git checkout -b fix/<short-topic>`
2. Stage changes (strictly exclude `.env*`, credentials, caches):
   ```bash
   git add <files>
   git commit -m "fix: <summary>"
   ```
3. Push branch:
   ```bash
   git push -u origin HEAD
   ```

### H2 — Create, approve & merge PR into `master`
1. Create PR into `master`:
   ```bash
   PR_URL=$(gh pr create --base master --title "fix: <summary>" --body "$(cat <<'EOF'
   ## Summary
   Hotfix: <summary>

   ## Changes
   - …

   ## Test plan
   - [x] Verified locally
   EOF
   )")
   ```
2. Attempt approval (ignore errors if GitHub disallows self-approval by author):
   ```bash
   gh pr review "$PR_URL" --approve -b "Approved hotfix" 2>/dev/null || true
   ```
3. Immediately merge and delete the remote branch:
   ```bash
   gh pr merge "$PR_URL" --squash --delete-branch --admin || gh pr merge "$PR_URL" --squash --delete-branch
   ```

### H3 — Cleanup & sync `master`
1. Switch to `master` and pull the newly merged commit:
   ```bash
   git fetch origin master
   git checkout master
   git pull origin master
   ```
2. Delete the local fix branch:
   ```bash
   git branch -D fix/<short-topic> 2>/dev/null || true
   ```
3. Report summary to user:
   - PR URL and squash merge status
   - Branch cleanup status
   - Clean git status on `master` (matching `origin/master`)

---

## Guardrails

- When invoked, execute branch creation, staging, commit, push, and `gh pr create` directly without asking for confirmation modals unless the user requested `draft only` / `dry run`.
- Immediately return the PR URL, title, and summary once `gh pr create` completes; do not block or poll for remote GitHub Actions CI to finish.
- In Hotfix mode: execute H1 through H3 consecutively without waiting for intermediate approvals.
- Never open a PR from `master` into `master` without a feature branch.
- Never delete or force-update the `master` branch.
- In Cleanup: do not run `reset --hard` or `branch -D` unless explicitly asked.
