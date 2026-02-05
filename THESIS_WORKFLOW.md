# 🎓 Thesis Project – Standard Operating Procedure (SOP)

**Project:** IT Ticket Classification
**Environment:** Azure ML · Conda · Poetry · GitHub

---

## 🟢 Phase 1: Daily Startup & Environment Alignment

Run these steps every time you start working to ensure hardware, environment, and dependencies are aligned.

### Start Azure Compute
Power on your Compute Instance via Azure ML Studio.

### Connect VS Code
Open VS Code and use the Azure extension to connect to your Compute Instance.

### Navigate to Repository & Activate Environment
```bash
cd Users/alsei/Thesis_IT_TicketClassification
conda activate myvenv
```

### Sync Dependencies (Source of Truth = poetry.lock)
```bash
poetry install
```

---

## 🔵 Phase 2: Synchronizing with the Team

Always sync before writing new code.

### Switch to Main
```bash
git checkout main
```

### Pull Latest Changes
```bash
git pull origin main --no-rebase
```

If the Nano editor opens for a merge message:
Press Ctrl+O, Enter, then Ctrl+X.

---

## 🟡 Phase 3: Development & Feature Work

Never code directly on main.

### Create a Feature Branch
```bash
git checkout -b feature/your-task-name
```

### Manage Packages (Example: spaCy)
```bash
poetry add spacy
poetry run python -m spacy download en_core_web_sm
```

### Execute Code
```bash
poetry run python your_script.py
```

---

## 🔴 Phase 4: Saving, Pushing & Merging

### Stage & Commit
```bash
git add .
git commit -m "feat: implemented NER extraction for ticket bodies"
```
If the commit fails and you see messages like:
- trailing-whitespace
- fix end of files
- files were modified by this hook

Then simply run:

```bash
git add .
git commit -m "feat: implemented NER extraction for ticket bodies"
```

### Push Branch
```bash
git push origin feature/your-task-name
```

### Create Pull Request
Go to GitHub, click “Compare & pull request”, and assign your friend as Reviewer.

---

## ⚪ Phase 5: Post-Merge Housekeeping

### Update Local Main
```bash
git fetch
git checkout main
git pull origin main
```

### Delete Old Feature Branch
```bash
git branch -d feature/your-task-name
```

---

## 🟠 Phase 6: Resolving Merge Conflicts

Merge conflicts happen when two people edit the same lines in the same file.

When Git reports a conflict, open the file and look for markers like:

```text
<<<<<<< HEAD
your version
=======
your friend's version
>>>>>>> main
```

Edit the file to keep the correct content and remove all markers.

Then mark the conflict as resolved:

```bash
git add path/to/file.py
git commit -m "fix: resolve merge conflict"
```

---

## ✅ Golden Rules

- main is stable and protected
- One task per feature branch
- poetry.lock is the source of truth
- Pull before coding
- PRs are mandatory
- Merge conflicts are normal

---

End of SOP
