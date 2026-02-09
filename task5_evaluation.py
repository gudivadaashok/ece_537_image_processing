"""
Task 5: Objective Evaluation (15%)
Compute metrics before vs after enhancement
- Sharpness proxy: Variance of Laplacian
- Global contrast: Standard deviation of luminance
- Shadow visibility: Fraction of pixels below threshold
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ObjectiveEvaluator:
    """Compute objective metrics for image enhancement evaluation."""

    def __init__(self, shadow_threshold=20):
        """
        Initialize evaluator.

        Parameters:
        -----------
        shadow_threshold : int
            Pixel value threshold for shadow visibility metric (default: 20)
        """
        self.shadow_threshold = shadow_threshold
        self.metrics = {}

    def compute_laplacian_variance(self, image, name=""):
        """
        Compute sharpness proxy: variance of Laplacian.

        The Laplacian is a second-order derivative operator that measures
        high-frequency content (edges). High variance indicates sharp edges.

        S = Var(∇²I)

        Parameters:
        -----------
        image : np.ndarray
            Input image (RGB or grayscale)
        name : str
            Name for this metric (for tracking)

        Returns:
        --------
        float
            Variance of Laplacian
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Compute Laplacian (second derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Compute variance
        sharpness = np.var(laplacian)

        if name:
            self.metrics[f"Sharpness - {name}"] = sharpness

        return sharpness

    def compute_global_contrast(self, image, name=""):
        """
        Compute global contrast: standard deviation of luminance.

        Measures the spread of brightness values. Higher std indicates
        better contrast and visibility of details.

        Parameters:
        -----------
        image : np.ndarray
            Input image (RGB or grayscale)
        name : str
            Name for this metric (for tracking)

        Returns:
        --------
        float
            Standard deviation of luminance
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Compute standard deviation
        contrast = np.std(gray)

        if name:
            self.metrics[f"Global Contrast - {name}"] = contrast

        return contrast

    def compute_shadow_visibility(self, image, name=""):
        """
        Compute shadow visibility: fraction of pixels below threshold.

        Measures what percentage of the image is still in deep shadow.
        Lower values indicate better shadow lifting.

        Parameters:
        -----------
        image : np.ndarray
            Input image (RGB or grayscale)
        name : str
            Name for this metric (for tracking)

        Returns:
        --------
        float
            Fraction of pixels below shadow_threshold (0.0 to 1.0)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Count pixels below threshold
        shadow_pixels = np.sum(gray < self.shadow_threshold)
        total_pixels = gray.size
        visibility = shadow_pixels / total_pixels

        if name:
            self.metrics[f"Shadow Fraction - {name}"] = visibility

        return visibility

    def compute_overexposure_fraction(self, image, threshold=235, name=""):
        """
        Compute overexposure fraction: fraction of pixels above threshold.

        Measures what percentage of the image is blown out (saturated).
        Lower values indicate better handling of bright areas.

        Parameters:
        -----------
        image : np.ndarray
            Input image (RGB or grayscale)
        threshold : int
            Pixel value threshold for overexposure (default: 235)
        name : str
            Name for this metric (for tracking)

        Returns:
        --------
        float
            Fraction of pixels above threshold (0.0 to 1.0)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Count pixels above threshold
        over_pixels = np.sum(gray > threshold)
        total_pixels = gray.size
        overexposure = over_pixels / total_pixels

        if name:
            self.metrics[f"Overexposure Fraction - {name}"] = overexposure

        return overexposure

    def evaluate_pair(self, before_image, after_image):
        """
        Compute all metrics for a before/after image pair.

        Parameters:
        -----------
        before_image : np.ndarray
            Original image (RGB)
        after_image : np.ndarray
            Enhanced image (RGB)

        Returns:
        --------
        dict
            Dictionary of all computed metrics
        """
        self.metrics = {}

        # Compute sharpness
        self.compute_laplacian_variance(before_image, "Before")
        self.compute_laplacian_variance(after_image, "After")

        # Compute contrast
        self.compute_global_contrast(before_image, "Before")
        self.compute_global_contrast(after_image, "After")

        # Compute shadow visibility
        self.compute_shadow_visibility(before_image, "Before")
        self.compute_shadow_visibility(after_image, "After")

        # Compute overexposure
        self.compute_overexposure_fraction(before_image, name="Before")
        self.compute_overexposure_fraction(after_image, name="After")

        return self.metrics

    def print_report(self):
        """Print formatted metric report."""
        print("\n" + "="*60)
        print("OBJECTIVE EVALUATION REPORT")
        print("="*60)

        # Group metrics by type
        metrics_by_type = {}
        for metric_name, value in self.metrics.items():
            metric_type = metric_name.split(" - ")[0]
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = {}
            metrics_by_type[metric_type][metric_name] = value

        # Print grouped metrics with improvements
        for metric_type in ["Sharpness", "Global Contrast", "Shadow Fraction", "Overexposure Fraction"]:
            if metric_type in metrics_by_type:
                print(f"\n{metric_type}:")
                before_val = metrics_by_type[metric_type].get(f"{metric_type} - Before")
                after_val = metrics_by_type[metric_type].get(f"{metric_type} - After")

                if before_val is not None and after_val is not None:
                    print(f"  Before: {before_val:.6f}")
                    print(f"  After:  {after_val:.6f}")

                    if metric_type in ["Shadow Fraction", "Overexposure Fraction"]:
                        # Lower is better for these metrics
                        change = ((before_val - after_val) / before_val * 100) if before_val != 0 else 0
                        print(f"  Change: {change:+.2f}% (↓ lower is better)")
                    else:
                        # Higher is better for these metrics
                        change = ((after_val - before_val) / before_val * 100) if before_val != 0 else 0
                        print(f"  Change: {change:+.2f}% (↑ higher is better)")

        print("\n" + "="*60)

    def plot_metrics(self, figsize=(12, 8)):
        """
        Create visualization of metrics comparison.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        # Organize data for plotting
        metrics_by_type = {}
        for metric_name, value in self.metrics.items():
            metric_type = metric_name.split(" - ")[0]
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = {}
            metrics_by_type[metric_type][metric_name] = value

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Objective Evaluation Metrics: Before vs After", fontsize=14, fontweight='bold')

        metric_types = ["Sharpness", "Global Contrast", "Shadow Fraction", "Overexposure Fraction"]
        axes_flat = axes.flatten()

        for idx, metric_type in enumerate(metric_types):
            ax = axes_flat[idx]

            if metric_type in metrics_by_type:
                before_key = f"{metric_type} - Before"
                after_key = f"{metric_type} - After"

                before_val = metrics_by_type[metric_type].get(before_key, 0)
                after_val = metrics_by_type[metric_type].get(after_key, 0)

                # Create bar plot
                categories = ['Before', 'After']
                values = [before_val, after_val]
                colors = ['#ff6b6b', '#51cf66']

                bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}',
                           ha='center', va='bottom', fontweight='bold')

                ax.set_ylabel('Value', fontweight='bold')
                ax.set_title(metric_type, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig


def main():
    """Example usage of ObjectiveEvaluator."""

    # Define data paths
    data_dir = Path("assignments/assignment1/data")

    # Load images
    before_path = data_dir / "assignment1" / "77.jpg"
    after_path = data_dir / "77_enhanced.jpg"

    print(f"Loading images...")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")

    # Read images
    before_bgr = cv2.imread(str(before_path))
    after_bgr = cv2.imread(str(after_path))

    if before_bgr is None or after_bgr is None:
        print("Error: Could not load images. Check file paths.")
        return

    # Convert BGR to RGB
    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)

    # Create evaluator and compute metrics
    evaluator = ObjectiveEvaluator(shadow_threshold=20)
    metrics = evaluator.evaluate_pair(before_rgb, after_rgb)

    # Print report
    evaluator.print_report()

    # Create visualization
    fig = evaluator.plot_metrics()

    # Save figure
    output_dir = Path("assignments/assignment1/data")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "task5_metrics_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nMetrics plot saved to: data/task5_metrics_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()

