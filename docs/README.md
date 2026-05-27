# EEG-COMET Documentation

This folder contains the source files for the EEG-COMET documentation
website, hosted on GitHub Pages.

The live site is published at:

> **<https://eeg-comet.github.io/eeg-comet/>**

It is built with [Jekyll](https://jekyllrb.com/) and the
[Just the Docs](https://just-the-docs.com/) theme.

---

## Building locally

You need Ruby 3.0+ and Bundler installed.

```bash
cd docs
bundle install
bundle exec jekyll serve --livereload
```

Open <http://localhost:4000/eeg-comet/> in your browser. Live reload will
re-render pages as you edit Markdown files.

> The local site is intentionally served under the `/eeg-comet/`
> sub-path so that internal links resolve identically to the deployed
> site.

---

## Folder layout

```
docs/
├── _config.yml             # Jekyll configuration
├── _sass/
│   └── color_schemes/
│       └── comet.scss      # EEG-COMET color theme
├── assets/
│   └── images/
│       └── logo.png        # EEG-COMET logo
├── modules/                # Per-module reference pages
│   ├── index.md
│   ├── data-loading.md
│   ├── data-preparation.md
│   ├── data-selection.md
│   ├── cluster-validation.md
│   ├── clustering.md
│   ├── labeling.md
│   ├── backfitting.md
│   ├── feature-extraction.md
│   ├── statistical-analysis.md
│   └── source-localization.md
├── index.md                # Homepage
├── installation.md         # Installation guide
├── getting-started.md      # Step-by-step tutorial
├── parameters.md           # Parameters reference
├── faq.md                  # Frequently asked questions
├── Gemfile                 # Ruby dependencies
└── README.md               # This file
```

---

## Deployment

Deployment is fully automated by the GitHub Actions workflow at
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml). The
workflow:

1. Triggers on pushes to `stable` or `main` (only when files in `docs/`
   or the workflow itself change), and can also be run manually from the
   **Actions** tab.
2. Installs Ruby and the Jekyll gems declared in `docs/Gemfile`.
3. Builds the site with `bundle exec jekyll build --baseurl ...`,
   where the base path is supplied by `actions/configure-pages` so the
   site works correctly under `/eeg-comet/`.
4. Uploads the build as a Pages artifact and deploys it through
   `actions/deploy-pages`.

### One-time GitHub Pages setup

1. Open **Settings → Pages** in the repository.
2. Under **Build and deployment**, set the **Source** to
   **GitHub Actions**.
3. Save.

From that point on, the workflow handles every release.

---

## Writing conventions

- Each page starts with a YAML front matter block setting `title`,
  `layout`, `nav_order`, and `description`.
- Internal page links use Jekyll's `{% link %}` tag
  (e.g. `{% link installation.md %}`); this resolves the path *and*
  prepends the site `baseurl`, so it works for both local and deployed
  builds.
- Asset URLs use the `relative_url` filter
  (e.g. `{{ '/assets/images/logo.png' | relative_url }}`).
- Callouts use Just the Docs's
  [callout classes](https://just-the-docs.com/docs/configuration/#callouts):
  `note`, `tip`, `highlight`, `important`, and `warning`.

---

## Contributing

1. Edit the relevant Markdown files (or add new ones with appropriate
   front matter).
2. Preview locally with `bundle exec jekyll serve` to verify formatting,
   navigation, and internal links.
3. Open a pull request against the `stable` branch.

---

## License

This documentation is part of EEG-COMET and is distributed under the
**GNU General Public License v3.0**. See the
[`LICENSE`](../LICENSE) file at the repository root for the full text.
