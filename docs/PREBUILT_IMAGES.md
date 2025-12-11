# Pre-built Container Images

This document explains how EMPO uses pre-built Docker container images for **automated environments** like GitHub Codespaces, CI/CD pipelines, and AI coding assistants.

> **⚠️ For Active Development**: If you're doing active development on EMPO, use the standard workflow instead:
> ```bash
> make up    # Builds locally and mounts your code
> make test  # Runs tests with your local changes
> make shell # Opens a shell in the container
> ```
> This ensures your local code changes and branch are always reflected. Pre-built images are primarily for **ephemeral environments** where fast startup matters more than local modifications.

## When to Use Pre-built Images vs Local Build

| Scenario | Recommended Approach |
|----------|---------------------|
| Active development on a feature branch | `make up` (local build) |
| GitHub Codespaces | Pre-built image (auto-configured) |
| CI/CD pipelines | Pre-built image |
| Quick testing without Docker build time | Pre-built image |
| Working on Dockerfile or requirements changes | `make up` (local build) |

## Overview

EMPO automatically builds and publishes Docker images to GitHub Container Registry (GHCR) on every push to the `main` or `develop` branches. These pre-built images include all dependencies already installed, eliminating the need to rebuild from scratch.

## Available Images

Images are published to `ghcr.io/mensch72/empo` with the following tags:

| Tag | Description |
|-----|-------------|
| `main` | Latest image from main branch (recommended for stable use) |
| `develop` | Latest image from develop branch (for testing new features) |
| `sha-<commit>` | Specific commit SHA (for reproducibility) |

## Using Pre-built Images

### For GitHub Codespaces / Dev Containers

The repository includes a `.devcontainer/devcontainer.json` configuration that automatically uses the pre-built image from GHCR. When you open this repository in:

- **GitHub Codespaces**: Click "Code" → "Codespaces" → "Create codespace on main"
- **VS Code with Dev Containers**: Open the repository and select "Reopen in Container"
- **AI Coding Assistants** (GitHub Copilot Workspace, etc.): The devcontainer configuration is automatically detected

The pre-built image includes:
- Python 3.11 with all dependencies from `requirements.txt` and `requirements-dev.txt`
- Development tools: pytest, black, ruff, mypy, jupyter
- System dependencies: graphviz, ffmpeg, MPI, etc.

### For Local Development

You can pull and use the pre-built image directly:

```bash
# Pull the latest image
docker pull ghcr.io/mensch72/empo:main

# Run interactively with your local code mounted
# On Linux/macOS (bash/zsh):
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -e PYTHONPATH=/workspace/src:/workspace/vendor/multigrid \
  ghcr.io/mensch72/empo:main \
  bash

# On Windows (PowerShell):
docker run -it --rm `
  -v "${PWD}:/workspace" `
  -e PYTHONPATH=/workspace/src:/workspace/vendor/multigrid `
  ghcr.io/mensch72/empo:main `
  bash
```

Or use it with Docker Compose by editing `.env`:

```bash
# In .env, add:
EMPO_IMAGE=ghcr.io/mensch72/empo:main
```

Then modify your `docker-compose.yml` to use `${EMPO_IMAGE:-empo:dev}` as the image.

### For CI/CD Pipelines

Use the pre-built image in your CI workflows:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/mensch72/empo:main
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/ -v
```

## Benefits

1. **Faster Setup**: Skip the 5-10 minute initial build time
2. **Consistent Environment**: Same dependencies across all developers and CI
3. **Reproducibility**: Pin to specific commit SHAs for reproducible builds
4. **Reduced Bandwidth**: Images are cached and layers are deduplicated

## Image Build Process

Images are built automatically by the GitHub Actions workflow in `.github/workflows/docker-build.yml`:

1. On push to `main` or `develop` branches
2. Tests run first to ensure the image is valid
3. Image is built with `DEV_MODE=true` to include development dependencies
4. Image is pushed to GHCR with branch name and commit SHA tags
5. GitHub Actions cache is used for faster rebuilds

## Updating the Image

The image is automatically rebuilt when:
- Changes are pushed to `main` or `develop`
- `requirements.txt` or `requirements-dev.txt` are modified
- The `Dockerfile` is updated

To manually trigger a rebuild, use the "workflow_dispatch" trigger in GitHub Actions.

## Security Note

The pre-built images are public and stored in GitHub Container Registry. They do not contain:
- API keys or secrets
- Private code or data
- User credentials

Sensitive data should always be provided at runtime via environment variables.

## Troubleshooting

### Image Not Found

If the image hasn't been built yet for your branch, you can:

1. Trigger the workflow manually in GitHub Actions
2. Build locally with `make build`
3. Use the `main` branch image as a fallback

### Permission Denied (Private Repositories Only)

**Note:** The EMPO images are public and don't require authentication to pull.

If you're working with a private fork and encounter authentication errors:

```bash
# Authenticate with GitHub using a Personal Access Token
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

### Outdated Image

If the image is missing recent dependencies:

1. Check when the image was last built in GitHub Actions
2. Ensure your branch has the latest changes from `main`
3. Manually trigger a rebuild if needed
