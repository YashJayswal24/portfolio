# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is Yash Jayswal's personal academic/portfolio website, built on the **al-folio** Jekyll theme (Jekyll 4.x, Ruby, Liquid templating, SCSS, GitHub Pages deployment). Deployed to `https://YashJayswal24.github.io/portfolio`.

## Common commands

**Local development (Docker — recommended):**
```bash
docker compose pull
docker compose up
# site at http://localhost:8080, live-reloads on file changes (5-10s rebuild)
```

**Local development (native Ruby):**
```bash
bundle install
bundle exec jekyll serve
# site at http://localhost:4000
```

**Build:**
```bash
bundle exec jekyll build     # output in _site/
JEKYLL_ENV=production bundle exec jekyll build   # production build (used in CI/deploy)
```

**Formatting (required — CI enforces this via `.github/workflows/prettier.yml`):**
```bash
npx prettier . --write        # format everything
npx prettier . --check        # check only (what CI runs)
```
Formatting rules live in `.prettierrc` (uses `@shopify/prettier-plugin-liquid`, 150 print width). Excluded paths are in `.prettierignore` (minified assets, `_scripts/*`, generated `_data/citations.yml`, etc.).

**Manual deploy to `gh-pages` branch** (normally automatic via GitHub Actions on push to `main`):
```bash
bin/deploy
```
This script requires a clean working tree, builds production, purges unused CSS, force-pushes `gh-pages`, and prompts for confirmation — treat it as a destructive/manual-only operation.

There is no JS/Ruby test suite in this repo. "Tests" in CI are: Prettier formatting check, broken-link checks, axe accessibility checks, CodeQL, and Lighthouse — all run via GitHub Actions on push/PR, not locally invocable as a single command.

## Architecture

This is a **content-over-code** repository: almost all day-to-day changes are to Markdown/YAML/BibTeX content files, not to Liquid/SCSS theme code.

- `_config.yml` — single source of truth for site-wide settings (title/name, url/baseurl, feature toggles like `enable_darkmode`, `enable_masonry`, analytics, `scholar:` author-highlighting config, `giscus:` comments config). Most feature on/off switches live here as `enable_*` booleans.
- `_pages/` — top-level site pages (`about.md`, `cv.md`, `publications.md`, `projects.md`, `blog.md`, `repositories.md`, `teaching.md`, etc.), Markdown with frontmatter.
- `_posts/` — blog posts, filename pattern `YYYY-MM-DD-title.md`.
- `_projects/` — project pages, one Markdown file per project.
- `_news/` — short news/announcement snippets shown on the home page.
- `_bibliography/papers.bib` — publications in BibTeX, with al-folio-specific extra fields (`abstract`, `pdf`, `code`, `website`, `slides`, `poster`, `selected`). Rendered via `jekyll-scholar`.
- `_data/` — YAML data: `socials.yml` (social links), `coauthors.yml`, `venues.yml` (BibTeX venue abbreviations), `repositories.yml` (GitHub repos shown on the repositories page), `citations.yml` (**generated** by `bin/update_scholar_citations.py` / `update-citations.yml` workflow — do not hand-edit).
- **CV/resume data**: the active source of truth is `assets/json/resume.json` (JSON Resume format), *not* `_data/cv.yml`. The CV layout (`_layouts/cv.liquid`) prefers the JSON file when present — editing `_data/cv.yml` has no effect while `resume.json` exists.
- `_layouts/` + `_includes/` — Liquid templates for the theme itself; only touch these for structural/theme changes, not routine content updates.
- `_sass/_themes.scss` / `_sass/_variables.scss` — theme color definitions (`--global-theme-color`); `_sass/_base.scss`, `_layout.scss`, `_cv.scss`, etc. for other styling.
- `_plugins/` — custom Jekyll plugins (Ruby) for citation counts, external post embedding, accent stripping, etc.
- `.agent/workflows/` — role-specific instruction docs for other agent personas already defined for this repo (`customize.agent.md` + `.extension.md` for site customization help, `docs.agent.md` for documentation-only edits, `deepml-agent.md` for a separate workflow that mirrors Deep-ML problem solutions from a sibling `deepml_solns` repo into `_posts/` blog entries tagged `categories: achievements`). Consult these when a task matches their described scope.
- `docker-compose.yml` pulls a prebuilt image (`amirpourmand/al-folio`) rather than building from the local `Dockerfile` by default.
- Deployment is automatic via `.github/workflows/deploy.yml` on push to `main`/`master` (builds with Jekyll, purges unused CSS with `purgecss`, publishes `_site/` via `github-pages-deploy-action`). `bin/deploy` is a legacy manual alternative that pushes directly to `gh-pages`.

## Content conventions

- Blog posts: `_posts/YYYY-MM-DD-descriptive-title.md`, frontmatter needs `layout: post`, `title`, `date`, `description`, `tags`, `categories`.
- Projects: `_projects/descriptive-name.md`, frontmatter needs `layout: post`... (`title`, `description`, `img`, `importance`).
- BibTeX entries go in `_bibliography/papers.bib`; author name highlighting is controlled by `scholar:first_name`/`scholar:last_name` in `_config.yml`.
- Images in `assets/img/`, PDFs in `assets/pdf/`.
