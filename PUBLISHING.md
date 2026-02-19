# Publishing To GitHub

Use `kmodel-core/` as the repository root for your portfolio project.

## Option 1: GitHub CLI

```bash
cd kmodel-core
git init
git add .
git commit -m "Initial commit: core first-kills ML pipeline"
gh repo create kmodel-core --public --source=. --remote=origin --push
```

## Option 2: Manual Remote Setup

```bash
cd kmodel-core
git init
git add .
git commit -m "Initial commit: core first-kills ML pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/kmodel-core.git
git push -u origin main
```

## Pre-Push Check

```bash
python -m pytest -q
```
