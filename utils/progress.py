"""
Progress tracking and time estimation for data generation.
Provides tqdm-based progress bars with ETA calculations.
"""

import time
from typing import Optional, Any
from tqdm import tqdm
from contextlib import contextmanager


class ProgressTracker:
    """
    Tracks progress across multiple stages of data generation.
    Provides progress bars and time estimates.
    """

    def __init__(self, enabled: bool = True, show_eta: bool = True):
        """
        Initialize the progress tracker.

        Args:
            enabled: Whether to show progress bars
            show_eta: Whether to show ETA estimates
        """
        self.enabled = enabled
        self.show_eta = show_eta
        self.stage_times = {}
        self.total_start_time = None
        self.current_stage = None

    def start(self):
        """Start tracking overall generation time."""
        self.total_start_time = time.time()

    def create_progress_bar(self, total: int, desc: str, unit: str = "rows") -> Optional[tqdm]:
        """
        Create a progress bar for a generation stage.

        Args:
            total: Total number of items to process
            desc: Description of the stage
            unit: Unit name for items (default: "rows")

        Returns:
            tqdm progress bar or None if disabled
        """
        if not self.enabled:
            return None

        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100
        )

    @contextmanager
    def stage(self, name: str):
        """
        Context manager for tracking a generation stage.

        Args:
            name: Name of the stage

        Usage:
            with tracker.stage("Generate Members"):
                # ... generation code ...
        """
        self.current_stage = name
        start_time = time.time()

        if self.enabled:
            print(f"\n{'=' * 80}")
            print(f"Starting: {name}")
            print(f"{'=' * 80}")

        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.stage_times[name] = elapsed

            if self.enabled:
                print(f"\n✓ Completed: {name} ({self._format_time(elapsed)})")

    def get_elapsed_time(self) -> float:
        """
        Get total elapsed time since start.

        Returns:
            Elapsed time in seconds
        """
        if self.total_start_time is None:
            return 0.0
        return time.time() - self.total_start_time

    def get_stage_time(self, stage_name: str) -> Optional[float]:
        """
        Get elapsed time for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Elapsed time in seconds, or None if stage not found
        """
        return self.stage_times.get(stage_name)

    def print_summary(self):
        """Print a summary of all stage timings."""
        if not self.enabled or not self.stage_times:
            return

        total_time = self.get_elapsed_time()

        print(f"\n{'=' * 80}")
        print("GENERATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"\nTotal Time: {self._format_time(total_time)}\n")
        print("Stage Breakdown:")
        print(f"{'Stage':<50} {'Time':<15} {'Percent':<10}")
        print("-" * 80)

        for stage_name, stage_time in self.stage_times.items():
            percent = (stage_time / total_time * 100) if total_time > 0 else 0
            print(f"{stage_name:<50} {self._format_time(stage_time):<15} {percent:>6.1f}%")

        print(f"{'=' * 80}\n")

    def _format_time(self, seconds: float) -> str:
        """
        Format time in human-readable format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


class BatchProgressTracker:
    """
    Tracks progress for batch processing operations.
    Useful for processing large datasets in chunks.
    """

    def __init__(self, total_items: int, batch_size: int, desc: str = "Processing", enabled: bool = True):
        """
        Initialize batch progress tracker.

        Args:
            total_items: Total number of items to process
            batch_size: Size of each batch
            desc: Description for progress bar
            enabled: Whether to show progress
        """
        self.total_items = total_items
        self.batch_size = batch_size
        self.enabled = enabled
        self.num_batches = (total_items + batch_size - 1) // batch_size

        self.pbar = None
        if enabled:
            self.pbar = tqdm(
                total=total_items,
                desc=desc,
                unit="items",
                unit_scale=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=100
            )

    def update(self, count: int):
        """
        Update progress by the specified count.

        Args:
            count: Number of items processed
        """
        if self.pbar:
            self.pbar.update(count)

    def close(self):
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def estimate_generation_time(n_members: int, claims_per_member: float = 4.0) -> dict:
    """
    Estimate total generation time based on dataset size.

    Args:
        n_members: Number of members to generate
        claims_per_member: Average claims per member

    Returns:
        Dictionary with estimated times per stage
    """
    # Empirical rates (rows per second) - these are rough estimates
    rates = {
        'insurance': 1000,      # Fast - only 8 rows
        'plans': 500,           # Fast - only 40 rows
        'facilities': 5000,     # Medium
        'conditions': 1000,     # Fast - only 50 rows
        'members': 2000,        # Slowest - complex calculations
        'enrollments': 3000,    # Medium
        'member_conditions': 4000,  # Fast
        'claims': 5000,         # Fast
    }

    # Calculate row counts
    n_facilities = max(int(n_members * 0.002), 50)
    n_enrollments = int(n_members * 0.90)
    n_member_conditions = int(n_members * 1.5)
    n_claims = int(n_members * claims_per_member)

    # Calculate estimated times
    estimates = {
        'insurance': 8 / rates['insurance'],
        'plans': 40 / rates['plans'],
        'facilities': n_facilities / rates['facilities'],
        'conditions': 50 / rates['conditions'],
        'members': n_members / rates['members'],
        'enrollments': n_enrollments / rates['enrollments'],
        'member_conditions': n_member_conditions / rates['member_conditions'],
        'claims': n_claims / rates['claims'],
    }

    total_time = sum(estimates.values())
    estimates['total'] = total_time
    estimates['total_formatted'] = _format_estimate(total_time)

    return estimates


def _format_estimate(seconds: float) -> str:
    """Format estimated time."""
    if seconds < 60:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        return f"~{int(seconds / 60)} minutes"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"~{hours}h {mins}m"


def print_generation_plan(config: dict, distributions: dict):
    """
    Print a summary of the generation plan.

    Args:
        config: Configuration dictionary
        distributions: Distributions dictionary
    """
    n_members = config.get('n_members', 100000)
    estimates = estimate_generation_time(n_members)

    print("\n" + "=" * 80)
    print("DATA GENERATION PLAN")
    print("=" * 80)
    print(f"\nDataset Scale:")
    print(f"  • Members: {n_members:,}")
    print(f"  • Facilities: ~{int(n_members * 0.002):,}")
    print(f"  • Enrollments: ~{int(n_members * 0.90):,}")
    print(f"  • Conditions: ~{int(n_members * 1.5):,}")
    print(f"  • Claims: ~{int(n_members * 4):,}")

    print(f"\nConfiguration:")
    print(f"  • Random seed: {config.get('random_seed', 'None (random)')}")
    print(f"  • Batch size: {config.get('batch_size', 25000):,}")
    print(f"  • Reference date: {config.get('reference_date', '2024-01-01')}")

    print(f"\nEstimated Generation Time: {estimates['total_formatted']}")
    print(f"{'=' * 80}\n")
