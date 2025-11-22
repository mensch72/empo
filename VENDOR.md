# Vendored Dependencies

This document describes the vendored (bundled) dependencies in this repository and how to manage them.

## Multigrid

We vendor the Multigrid library source code to allow local modifications without rebuilding the Docker container.

### Location

The Multigrid source code is located in:
```
vendor/multigrid/
```

The main Python package is at:
```
vendor/multigrid/gym_multigrid/
```

### How It Works

Unlike traditional pip installation, the vendored Multigrid is imported via **PYTHONPATH**:

- The Dockerfile sets: `PYTHONPATH=/workspace:/workspace/vendor/multigrid`
- This allows Python to import `gym_multigrid` directly from the vendored source
- **No container rebuild needed** when you modify the source code
- Changes are immediately reflected when you restart Python or re-import the module

### Making Local Modifications

You can freely modify the source code in `vendor/multigrid/`. Changes take effect immediately without rebuilding the container.

**Common files to modify:**
- `vendor/multigrid/gym_multigrid/envs/` - Environment definitions
- `vendor/multigrid/gym_multigrid/multigrid.py` - Core multigrid classes
- `vendor/multigrid/gym_multigrid/rendering.py` - Rendering utilities

**Workflow:**
1. Edit files in `vendor/multigrid/gym_multigrid/`
2. Restart your Python interpreter or re-import the module
3. Changes are immediately available (no rebuild required)

### For Local Development (Outside Docker)

If you want to use the vendored Multigrid outside of Docker:

```bash
# Option 1: Set PYTHONPATH
export PYTHONPATH=/path/to/empo:/path/to/empo/vendor/multigrid:$PYTHONPATH

# Option 2: Install in editable mode
pip install -e ./vendor/multigrid
```

### Pulling Upstream Updates

To pull the latest changes from the upstream Multigrid repository:

```bash
# Pull updates from upstream
git subtree pull --prefix=vendor/multigrid https://github.com/ArnaudFickinger/gym-multigrid.git master --squash
```

**Important Notes:**
- The `--squash` flag combines all upstream commits into a single commit
- If you have local modifications, there may be merge conflicts that need to be resolved
- Always test after pulling upstream updates to ensure compatibility
- No container rebuild needed after pulling updates

### Pushing Local Changes Upstream (Optional)

If you want to contribute your local changes back to the Multigrid project:

1. Create a separate branch with your changes:
   ```bash
   git subtree split --prefix=vendor/multigrid -b multigrid-changes
   ```

2. Push to your fork of Multigrid:
   ```bash
   git push your-multigrid-fork multigrid-changes:feature-branch
   ```

3. Create a pull request on the upstream Multigrid repository

### Version Information

To check the current vendored version, look at the source code or check the git history:
```bash
git log --oneline vendor/multigrid/ | head -5
```

### Troubleshooting

**Issue: Import errors after modifying the source**
- Restart your Python interpreter or container
- Verify PYTHONPATH is set: `echo $PYTHONPATH`
- Check that `vendor/multigrid` is in PYTHONPATH

**Issue: Changes not reflected**
- Ensure you restart Python or re-import the module
- Python caches imported modules - use `importlib.reload()` if needed
- For interactive development, consider using `%load_ext autoreload` in Jupyter

**Issue: Merge conflicts when pulling updates**
- Resolve conflicts in `vendor/multigrid/` like normal git conflicts
- Test thoroughly after resolution
- Consider stashing your changes, pulling updates, then reapplying

**Issue: Want to revert to a specific upstream version**
```bash
# Check available tags/commits
git ls-remote https://github.com/ArnaudFickinger/gym-multigrid.git

# Pull a specific commit
git subtree pull --prefix=vendor/multigrid https://github.com/ArnaudFickinger/gym-multigrid.git <commit-hash> --squash
```

## Why Use PYTHONPATH Instead of pip install?

For actively developed dependencies like Multigrid, using PYTHONPATH provides several advantages:

- **No rebuild required**: Edit code and immediately see changes
- **Fast iteration**: Perfect for making extensive modifications
- **Simple workflow**: Just edit files and restart Python
- **Docker-friendly**: Changes persist across container restarts via bind mount
- **Same as development**: Identical behavior in Docker and local development

## Alternative: pip install -e (Editable Install)

If you prefer traditional editable installs, you can modify the Dockerfile:

```dockerfile
# Copy vendored dependencies
COPY vendor /tmp/vendor

# Install in editable mode
RUN pip install -e /tmp/vendor/multigrid
```

However, this requires:
- Rebuilding the container when switching between modified/unmodified versions
- More complex workflow
- Not recommended for extensive changes

## Why Use Git Subtree?

Git subtree was chosen over alternatives for these reasons:

- **No submodule complexity**: No need for `git submodule update --init`
- **Full source in repo**: All code is in the main repository
- **Easy for collaborators**: No special git commands needed for basic usage
- **Flexible modifications**: Can modify vendored code freely
- **Upstream sync**: Can pull updates from upstream when needed

## Upstream Repository

- **Repository**: https://github.com/ArnaudFickinger/gym-multigrid
- **License**: Apache 2.0 (see vendor/multigrid/LICENSE)
- **Description**: Multi-agent gridworld environments for reinforcement learning
