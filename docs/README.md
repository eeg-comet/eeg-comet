# EEG-COMET Documentation

This folder contains the source files for the EEG-COMET documentation website, hosted on GitHub Pages.

## Viewing the Documentation

The documentation is available at: **https://eeg-comet.github.io**

## Building Locally

To build and preview the documentation locally:

### Prerequisites

1. Install Ruby (2.7+)
2. Install Bundler: `gem install bundler`

### Build Steps

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000/ in your browser.

## Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── _sass/               # Custom SCSS styles
│   └── color_schemes/
│       └── comet.scss   # EEG-COMET color theme
├── assets/
│   └── images/
│       └── logo.png     # EEG-COMET logo
├── modules/             # Module documentation
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
├── index.md             # Homepage
├── installation.md      # Installation guide
├── getting-started.md   # Getting started tutorial
├── parameters.md        # Parameters reference
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## Deploying to GitHub Pages

The documentation is automatically deployed when pushed to the `main` branch.

### Setup GitHub Pages

1. Go to repository Settings → Pages
2. Under "Source", select "Deploy from a branch"
3. Select `main` branch and `/docs` folder
4. Save

The site will be built and deployed automatically.

## Theme

The documentation uses the [Just the Docs](https://just-the-docs.com/) theme with custom EEG-COMET colors inspired by the comet logo.

## Contributing

To contribute to the documentation:

1. Edit the relevant Markdown files
2. Preview locally using Jekyll
3. Submit a pull request

## License

This documentation is part of EEG-COMET and is distributed under the **GNU General Public License v3.0**. See the [`LICENSE`](../LICENSE) file at the repository root for the full text.

