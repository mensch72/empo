# Video Capture Guide for EMPO Environments

This guide explains how to capture videos from EMPO environments, covering both MultiGrid and Transport environments.

## Overview

EMPO provides two main environment types with different video capture approaches:

1. **MultiGrid Environments** - Grid-based environments with tile-based rendering
2. **Transport Environments** - Graph-based transport networks with vehicle routing

## Transport Environment Video Capture

The Transport environment (`ai_transport`) has built-in video recording capabilities that should be used instead of custom rendering code.

### Basic Pattern

```python
from ai_transport import parallel_env

# Create environment
env = parallel_env(...)

# Start video recording
env.unwrapped.start_video_recording()

# Reset environment
observations = env.reset()

# Render initial state
env.unwrapped.render(
    goal_info=...,  # Optional: dict mapping agent to goal node
    title="Initial State"
)

# Run simulation loop
for step in range(num_steps):
    # Get actions
    actions = {...}
    
    # Execute step
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # IMPORTANT: Render after each step to capture frame
    env.unwrapped.render(
        goal_info=...,
        title=f"Step {step}"
    )

# Save video
env.unwrapped.save_video("output.mp4", fps=2)
env.close()
```

### Key Points

1. **Always use `env.unwrapped`**: When `render_mode='human'` is set during environment creation, use `env.unwrapped` to access the actual environment instance for video methods.

2. **Render inside the loop**: You must call `env.unwrapped.render()` after each `env.step()`. Missing this call means no frames are captured, resulting in a video with identical frames.

3. **Video format**: The `save_video()` method tries to save as MP4 using matplotlib's FFMpegWriter. If ffmpeg is not available, it automatically falls back to GIF using the pillow writer.

4. **Frame rate**: The `fps` parameter in `save_video()` controls the video playback speed. Typical values are 1-5 fps for simulation videos.

### Common Mistakes

#### ❌ Missing render() in loop

```python
env.unwrapped.start_video_recording()
env.unwrapped.render(title="Initial")  # Only renders once!

for step in range(100):
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # Missing render() call here!

env.unwrapped.save_video("output.mp4")  # Video has 1 frame repeated 100 times
```

#### ✅ Correct pattern

```python
env.unwrapped.start_video_recording()
env.unwrapped.render(title="Initial")

for step in range(100):
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.unwrapped.render(title=f"Step {step}")  # Captures each state

env.unwrapped.save_video("output.mp4")  # Video shows actual progression
```

### Continuous Movement Visualization

The Transport environment automatically renders continuous movement:

- **Multi-frame rendering**: The `render()` method generates multiple frames proportional to simulation time elapsed (typically 2 frames per second of simulation time)
- **Interpolated positions**: When agents move along edges, they are rendered at interpolated positions showing smooth movement
- **Substeps**: Movement actions are broken into 5 substeps with intermediate rendering

This means a single `render()` call after each `env.step()` captures smooth continuous movement - you don't need to manually interpolate frames.

### Visualization Features

The Transport environment's rendering includes:

- **Bidirectional lanes**: Roads shown as separate lanes with arrows indicating direction
- **Vehicle rectangles**: Vehicles rendered as rectangles sized by capacity, rotated to align with road direction
- **Passengers inside vehicles**: Humans aboard vehicles shown as smaller red dots inside vehicle rectangles
- **Vehicle destinations**: Curved blue dotted arcs from vehicles to their announced destinations
- **Goal markers**: Red stars showing human goal nodes
- **Cluster information**: Visual distinction between node clusters

## MultiGrid Environment Video Capture

MultiGrid environments use a different approach based on matplotlib animation or PIL image sequences.

### Basic Pattern

```python
from pettingzoo.utils import parallel_to_aec
from src.empo.examples.multiworld import create_env

# Create environment
env = create_env(...)

# Collect frames during rollout
frames = []

observations = env.reset()
rgb_array = env.render()
frames.append(rgb_array)

for step in range(num_steps):
    actions = {...}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    rgb_array = env.render()
    frames.append(rgb_array)

# Save video using matplotlib or PIL
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
img = ax.imshow(frames[0])

def update(frame):
    img.set_data(frames[frame])
    return [img]

anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
anim.save("output.mp4", writer='ffmpeg', fps=5)
plt.close()
```

## Error Handling

### "No frames recorded" Error

This error occurs when `save_video()` is called before `start_video_recording()` or when no `render()` calls were made:

```python
# Wrong
env.unwrapped.save_video("output.mp4")  # Error: No frames recorded

# Correct
env.unwrapped.start_video_recording()
env.unwrapped.render(title="Frame 1")
env.unwrapped.save_video("output.mp4")  # Works
```

### Using `env` vs `env.unwrapped`

Always use `env.unwrapped` for video methods to avoid issues:

```python
# Wrong - may cause buffer reuse issues
env.start_video_recording()
env.render(title="Frame 1")
env.save_video("output.mp4")

# Correct
env.unwrapped.start_video_recording()
env.unwrapped.render(title="Frame 1")
env.unwrapped.save_video("output.mp4")
```

If you try to use `env.render()` when you should use `env.unwrapped.render()`, you may get an exception explaining this requirement.

## Example Scripts

See the following example scripts for complete working examples:

1. **Transport Examples**:
   - `examples/transport/transport_two_cluster_demo.py` - Learning demo with training
   - `examples/transport/transport_handcrafted_demo.py` - Minimal hand-crafted demo
   - `examples/transport/transport_stress_test_demo.py` - Large-scale stress test

2. **MultiGrid Examples**:
   - `examples/multiworld_demo.py` - MultiGrid environment demo

## Troubleshooting

### Video shows no movement

**Problem**: All frames in the video are identical, only the title changes.

**Cause**: Missing `render()` call inside the simulation loop.

**Solution**: Add `env.unwrapped.render()` after each `env.step()`.

### Video is choppy or has gaps

**Problem**: Some frames are missing or movement appears discontinuous.

**Cause**: Inconsistent `render()` calls or skipping steps.

**Solution**: Ensure `render()` is called after every single `env.step()`.

### ImportError or ModuleNotFoundError

**Problem**: Missing dependencies for video encoding.

**Cause**: ffmpeg or required Python packages not installed.

**Solution**: 
- For MP4: Install ffmpeg system package
- For GIF fallback: Install pillow (`pip install pillow`)
- For matplotlib-based approaches: Install matplotlib (`pip install matplotlib`)

## Performance Considerations

- **Frame generation**: Each `render()` call may generate multiple frames (especially for Transport with continuous movement). This is normal and ensures smooth visualization.
- **File size**: MP4 files are typically smaller than GIFs. Use MP4 when possible.
- **Rendering speed**: Rendering can be slow for large networks with many agents. Consider reducing the number of steps for quick tests.
- **Memory usage**: Frames are stored in memory until `save_video()` is called. For very long rollouts (1000+ steps), consider periodic video saving.
