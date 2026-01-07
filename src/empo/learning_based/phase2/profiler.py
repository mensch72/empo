"""
Training profiler for Phase 2 training loop.

Provides detailed timing instrumentation for different training components:
- Action sampling (robot and human)
- Transition probability calculation
- Network training (RND, encoders, heads)
- Logging/reporting

Usage:
    # No overhead when disabled (default):
    profiler = NoOpProfiler()
    
    # Enable profiling:
    profiler = TrainingProfiler()
    
    # Use in code:
    with profiler.section("sampling_actions"):
        action = sample_action()
    
    # Get report:
    print(profiler.report())
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional


class NoOpProfiler:
    """
    Profiler that does nothing - minimal overhead.
    
    The context manager protocol adds ~50-100ns per call, which is
    negligible compared to tensor operations.
    """
    
    @contextmanager
    def section(self, name: str):
        """No-op context manager."""
        yield
    
    def start(self, name: str):
        """No-op start."""
    
    def stop(self, name: str):
        """No-op stop."""
    
    def step(self):
        """No-op step (compatibility with torch.profiler interface)."""
    
    def start_profiling(self):
        """No-op start profiling."""
    
    def stop_profiling(self):
        """No-op stop profiling."""
    
    def report(self) -> str:
        """Return empty report."""
        return ""
    
    def reset(self):
        """No-op reset."""
    
    def get_summary(self) -> Dict[str, float]:
        """Return empty summary."""
        return {}

    def save_report(self, output_dir: str, basename: str = "profiler_report") -> None:
        """No-op save."""


class TrainingProfiler:
    """
    Detailed profiler for Phase 2 training components.
    
    Categories tracked:
    - Actor operations:
        - sample_human_actions: Sampling from human policy prior
        - sample_robot_action: Sampling from robot Q-network policy
        - transition_probabilities: Computing env transition probs
        - step_environment: Stepping the environment
        - replay_buffer: Adding to replay buffer
        - goal_sampling: Sampling goals for humans
    
    - Learner operations:
        - batch_sampling: Sampling from replay buffer
        - forward_v_h_e: V_h^e network forward pass
        - forward_x_h: X_h network forward pass  
        - forward_u_r: U_r network forward pass
        - forward_q_r: Q_r network forward pass
        - forward_v_r: V_r network forward pass
        - forward_rnd: RND network forward pass
        - forward_human_rnd: Human action RND forward
        - target_computation: Computing TD targets
        - loss_computation: Computing losses
        - backward_pass: Gradient computation
        - optimizer_step: Optimizer updates
        - target_network_update: Soft/hard target updates
    
    - Logging operations:
        - tensorboard_logging: Writing to TensorBoard
        - progress_bar: Updating tqdm progress bar
    
    Example:
        profiler = TrainingProfiler()
        
        for step in range(num_steps):
            with profiler.section("sample_human_actions"):
                human_actions = sample_humans()
            
            with profiler.section("transition_probabilities"):
                trans_probs = env.transition_probabilities(state, actions)
            
            # ... more sections ...
        
        print(profiler.report())
    """
    
    # Category groupings for report
    CATEGORIES = {
        "Actor (env interaction)": [
            "sample_human_actions",
            "sample_robot_action", 
            "transition_probabilities",
            "step_environment",
            "replay_buffer",
            "goal_sampling",
            "curiosity_bonus",
            "actor_total",
        ],
        "Learner (network training)": [
            "batch_sampling",
            "forward_v_h_e",
            "forward_x_h",
            "forward_u_r", 
            "forward_q_r",
            "forward_v_r",
            "forward_rnd",
            "forward_human_rnd",
            "target_computation",
            "loss_computation",
            "backward_pass",
            "optimizer_step",
            "target_network_update",
            "learner_total",
        ],
        "Logging": [
            "tensorboard_logging",
            "progress_bar",
            "warmup_check",
            "logging_total",
        ],
    }
    
    def __init__(self):
        """Initialize profiler with empty counters."""
        self.times: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}
        self._total_time: float = 0.0
        self._profiling_start: Optional[float] = None
    
    @contextmanager
    def section(self, name: str):
        """
        Context manager to time a code section.
        
        Args:
            name: Section name for reporting.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name] += elapsed
            self.counts[name] += 1
    
    def start(self, name: str):
        """
        Start timing a section manually.
        
        Use with stop() for non-context-manager timing.
        
        Args:
            name: Section name.
        """
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str):
        """
        Stop timing a section started with start().
        
        Args:
            name: Section name (must match start()).
        """
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name] += elapsed
            self.counts[name] += 1
            del self._start_times[name]
    
    def start_profiling(self):
        """Mark the start of profiling (for total time calculation)."""
        self._profiling_start = time.perf_counter()
    
    def stop_profiling(self):
        """Mark the end of profiling (for total time calculation)."""
        if self._profiling_start is not None:
            self._total_time = time.perf_counter() - self._profiling_start
    
    def step(self):
        """Step marker (compatibility with torch.profiler interface)."""
    
    def reset(self):
        """Reset all counters."""
        self.times.clear()
        self.counts.clear()
        self._start_times.clear()
        self._total_time = 0.0
        self._profiling_start = None
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get timing summary as a dictionary.
        
        Returns:
            Dict mapping section name to total time in seconds.
        """
        return dict(self.times)
    
    def report(self) -> str:
        """
        Generate a detailed timing report.
        
        Returns:
            Formatted string with timing breakdown by category.
        """
        lines = []
        lines.append("")
        lines.append("=" * 90)
        lines.append("TRAINING PROFILER REPORT")
        lines.append("=" * 90)
        
        # Calculate totals
        total_measured = sum(self.times.values())
        
        # If we have total time, calculate unmeasured overhead
        if self._total_time > 0:
            overhead = self._total_time - total_measured
            overhead_pct = 100 * overhead / self._total_time if self._total_time > 0 else 0
        else:
            overhead = 0
            overhead_pct = 0
        
        # Header
        lines.append(f"{'Section':<40} {'Time (s)':<12} {'%':<8} {'Count':<10} {'Avg (ms)':<12}")
        lines.append("-" * 90)
        
        # Report by category
        for category, section_names in self.CATEGORIES.items():
            # Check if any sections in this category have data
            category_sections = [(n, self.times.get(n, 0), self.counts.get(n, 0)) 
                                for n in section_names if n in self.times]
            
            if not category_sections:
                continue
            
            # Category header
            lines.append(f"\n{category}:")
            lines.append("-" * 40)
            
            category_total = 0
            for name, t, count in sorted(category_sections, key=lambda x: -x[1]):
                category_total += t
                pct = 100 * t / total_measured if total_measured > 0 else 0
                avg_ms = 1000 * t / count if count > 0 else 0
                lines.append(f"  {name:<38} {t:>10.3f}s {pct:>7.1f}% {count:>10} {avg_ms:>10.3f}ms")
            
            # Category subtotal
            cat_pct = 100 * category_total / total_measured if total_measured > 0 else 0
            lines.append(f"  {'[Subtotal]':<38} {category_total:>10.3f}s {cat_pct:>7.1f}%")
        
        # Uncategorized sections
        all_categorized = set()
        for sections in self.CATEGORIES.values():
            all_categorized.update(sections)
        
        uncategorized = [(n, self.times[n], self.counts[n]) 
                        for n in self.times if n not in all_categorized]
        
        if uncategorized:
            lines.append(f"\nOther:")
            lines.append("-" * 40)
            for name, t, count in sorted(uncategorized, key=lambda x: -x[1]):
                pct = 100 * t / total_measured if total_measured > 0 else 0
                avg_ms = 1000 * t / count if count > 0 else 0
                lines.append(f"  {name:<38} {t:>10.3f}s {pct:>7.1f}% {count:>10} {avg_ms:>10.3f}ms")
        
        # Totals
        lines.append("")
        lines.append("=" * 90)
        lines.append(f"{'TOTAL MEASURED':<40} {total_measured:>10.3f}s {100.0:>7.1f}%")
        if self._total_time > 0:
            lines.append(f"{'UNMEASURED OVERHEAD':<40} {overhead:>10.3f}s {overhead_pct:>7.1f}%")
            lines.append(f"{'TOTAL WALL TIME':<40} {self._total_time:>10.3f}s")
        lines.append("=" * 90)
        
        return "\n".join(lines)
    
    def report_summary(self) -> str:
        """
        Generate a compact one-line summary.
        
        Returns:
            Single line with key timing percentages.
        """
        total = sum(self.times.values())
        if total == 0:
            return "No profiling data collected"
        
        # Aggregate by high-level category
        actor_time = sum(self.times.get(s, 0) for s in [
            "sample_human_actions", "sample_robot_action", "transition_probabilities",
            "step_environment", "replay_buffer", "goal_sampling", "curiosity_bonus"
        ])
        learner_time = sum(self.times.get(s, 0) for s in [
            "batch_sampling", "forward_v_h_e", "forward_x_h", "forward_u_r",
            "forward_q_r", "forward_v_r", "forward_rnd", "forward_human_rnd",
            "target_computation", "loss_computation", "backward_pass", 
            "optimizer_step", "target_network_update"
        ])
        logging_time = sum(self.times.get(s, 0) for s in [
            "tensorboard_logging", "progress_bar", "warmup_check"
        ])
        
        actor_pct = 100 * actor_time / total
        learner_pct = 100 * learner_time / total
        logging_pct = 100 * logging_time / total
        
        return (f"Actor: {actor_pct:.1f}% | Learner: {learner_pct:.1f}% | "
                f"Logging: {logging_pct:.1f}% | Total: {total:.2f}s")

    def save_report(self, output_dir: str, basename: str = "profiler_report") -> None:
        """
        Save profiling report to markdown and HTML files.
        
        Args:
            output_dir: Directory to save reports to.
            basename: Base filename (without extension).
        """
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate markdown report
        md_path = os.path.join(output_dir, f"{basename}.md")
        html_path = os.path.join(output_dir, f"{basename}.html")
        
        # Calculate totals
        total_measured = sum(self.times.values())
        
        # Markdown content
        md_lines = []
        md_lines.append("# Training Profiler Report")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        
        if self._total_time > 0:
            md_lines.append(f"**Total wall time:** {self._total_time:.2f}s")
            md_lines.append("")
        
        md_lines.append("## Summary")
        md_lines.append("")
        md_lines.append(self.report_summary())
        md_lines.append("")
        
        md_lines.append("## Detailed Breakdown")
        md_lines.append("")
        
        # Report by category
        for category, section_names in self.CATEGORIES.items():
            category_sections = [(n, self.times.get(n, 0), self.counts.get(n, 0)) 
                                for n in section_names if n in self.times]
            
            if not category_sections:
                continue
            
            md_lines.append(f"### {category}")
            md_lines.append("")
            md_lines.append("| Section | Time (s) | % | Count | Avg (ms) |")
            md_lines.append("|---------|----------|---|-------|----------|")
            
            category_total = 0
            for name, t, count in sorted(category_sections, key=lambda x: -x[1]):
                category_total += t
                pct = 100 * t / total_measured if total_measured > 0 else 0
                avg_ms = 1000 * t / count if count > 0 else 0
                md_lines.append(f"| {name} | {t:.3f} | {pct:.1f}% | {count} | {avg_ms:.3f} |")
            
            cat_pct = 100 * category_total / total_measured if total_measured > 0 else 0
            md_lines.append(f"| **Subtotal** | **{category_total:.3f}** | **{cat_pct:.1f}%** | | |")
            md_lines.append("")
        
        # Uncategorized sections
        all_categorized = set()
        for sections in self.CATEGORIES.values():
            all_categorized.update(sections)
        
        uncategorized = [(n, self.times[n], self.counts[n]) 
                        for n in self.times if n not in all_categorized]
        
        if uncategorized:
            md_lines.append("### Other")
            md_lines.append("")
            md_lines.append("| Section | Time (s) | % | Count | Avg (ms) |")
            md_lines.append("|---------|----------|---|-------|----------|")
            for name, t, count in sorted(uncategorized, key=lambda x: -x[1]):
                pct = 100 * t / total_measured if total_measured > 0 else 0
                avg_ms = 1000 * t / count if count > 0 else 0
                md_lines.append(f"| {name} | {t:.3f} | {pct:.1f}% | {count} | {avg_ms:.3f} |")
            md_lines.append("")
        
        # Totals
        md_lines.append("## Totals")
        md_lines.append("")
        md_lines.append(f"- **Total measured:** {total_measured:.3f}s")
        if self._total_time > 0:
            overhead = self._total_time - total_measured
            md_lines.append(f"- **Unmeasured overhead:** {overhead:.3f}s")
            md_lines.append(f"- **Total wall time:** {self._total_time:.3f}s")
        
        # Write markdown
        md_content = "\n".join(md_lines)
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        # Generate HTML with styling
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("<title>Training Profiler Report</title>")
        html_lines.append("<style>")
        html_lines.append("""
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
       max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
h2 { color: #444; margin-top: 30px; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; background: white; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }
th { background: #4CAF50; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
tr:hover { background: #f1f1f1; }
.subtotal { font-weight: bold; background: #e8f5e9 !important; }
.summary { background: white; padding: 15px; border-radius: 5px; 
           box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin: 15px 0; }
.meta { color: #666; font-size: 0.9em; }
.totals { background: white; padding: 15px; border-radius: 5px; 
          box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
""")
        html_lines.append("</style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("<h1>Training Profiler Report</h1>")
        html_lines.append(f'<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        if self._total_time > 0:
            html_lines.append(f'<p class="meta">Total wall time: {self._total_time:.2f}s</p>')
        
        html_lines.append("<h2>Summary</h2>")
        html_lines.append(f'<div class="summary">{self.report_summary()}</div>')
        
        html_lines.append("<h2>Detailed Breakdown</h2>")
        
        # Report by category
        for category, section_names in self.CATEGORIES.items():
            category_sections = [(n, self.times.get(n, 0), self.counts.get(n, 0)) 
                                for n in section_names if n in self.times]
            
            if not category_sections:
                continue
            
            html_lines.append(f"<h3>{category}</h3>")
            html_lines.append("<table>")
            html_lines.append("<tr><th>Section</th><th>Time (s)</th><th>%</th><th>Count</th><th>Avg (ms)</th></tr>")
            
            category_total = 0
            for name, t, count in sorted(category_sections, key=lambda x: -x[1]):
                category_total += t
                pct = 100 * t / total_measured if total_measured > 0 else 0
                avg_ms = 1000 * t / count if count > 0 else 0
                html_lines.append(f"<tr><td>{name}</td><td>{t:.3f}</td><td>{pct:.1f}%</td><td>{count}</td><td>{avg_ms:.3f}</td></tr>")
            
            cat_pct = 100 * category_total / total_measured if total_measured > 0 else 0
            html_lines.append(f'<tr class="subtotal"><td>Subtotal</td><td>{category_total:.3f}</td><td>{cat_pct:.1f}%</td><td></td><td></td></tr>')
            html_lines.append("</table>")
        
        # Uncategorized
        if uncategorized:
            html_lines.append("<h3>Other</h3>")
            html_lines.append("<table>")
            html_lines.append("<tr><th>Section</th><th>Time (s)</th><th>%</th><th>Count</th><th>Avg (ms)</th></tr>")
            for name, t, count in sorted(uncategorized, key=lambda x: -x[1]):
                pct = 100 * t / total_measured if total_measured > 0 else 0
                avg_ms = 1000 * t / count if count > 0 else 0
                html_lines.append(f"<tr><td>{name}</td><td>{t:.3f}</td><td>{pct:.1f}%</td><td>{count}</td><td>{avg_ms:.3f}</td></tr>")
            html_lines.append("</table>")
        
        # Totals
        html_lines.append("<h2>Totals</h2>")
        html_lines.append('<div class="totals">')
        html_lines.append(f"<p><strong>Total measured:</strong> {total_measured:.3f}s</p>")
        if self._total_time > 0:
            overhead = self._total_time - total_measured
            html_lines.append(f"<p><strong>Unmeasured overhead:</strong> {overhead:.3f}s</p>")
            html_lines.append(f"<p><strong>Total wall time:</strong> {self._total_time:.3f}s</p>")
        html_lines.append("</div>")
        
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        # Write HTML
        with open(html_path, 'w') as f:
            f.write("\n".join(html_lines))
        
        print(f"Profiler reports saved to:")
        print(f"  Markdown: {md_path}")
        print(f"  HTML: {html_path}")
