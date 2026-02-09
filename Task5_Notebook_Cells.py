# Task 5: Objective Evaluation (15%)
# Compute metrics before vs after enhancement

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# %% md
## 5.1 Metric Functions

def compute_laplacian_variance(image):
    """
    Compute sharpness proxy: variance of Laplacian.

    The Laplacian is a second-order derivative operator that measures
    high-frequency content (edges). High variance indicates sharp edges.

    S = Var(∇²I)

    Args:
        image: Input image (RGB or grayscale)

    Returns:
        float: Variance of Laplacian
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

    return sharpness


def compute_global_contrast(image):
    """
    Compute global contrast: standard deviation of luminance.

    Measures the spread of brightness values. Higher std indicates
    better contrast and visibility of details.

    Args:
        image: Input image (RGB or grayscale)

    Returns:
        float: Standard deviation of luminance
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    # Compute standard deviation
    contrast = np.std(gray)

    return contrast


def compute_shadow_visibility(image, threshold=20):
    """
    Compute shadow visibility: fraction of pixels below threshold.

    Measures what percentage of the image is still in deep shadow.
    Lower values indicate better shadow lifting.

    Args:
        image: Input image (RGB or grayscale)
        threshold: Pixel value threshold (default: 20)

    Returns:
        float: Fraction of pixels below threshold (0.0 to 1.0)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    # Count pixels below threshold
    shadow_pixels = np.sum(gray < threshold)
    total_pixels = gray.size
    visibility = shadow_pixels / total_pixels

    return visibility


def compute_overexposure_fraction(image, threshold=235):
    """
    Compute overexposure fraction: fraction of pixels above threshold.

    Measures what percentage of the image is blown out (saturated).
    Lower values indicate better handling of bright areas.

    Args:
        image: Input image (RGB or grayscale)
        threshold: Pixel value threshold for overexposure (default: 235)

    Returns:
        float: Fraction of pixels above threshold (0.0 to 1.0)
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

    return overexposure


# %% md
## 5.2 Load Images and Compute Metrics

# Load images
before_bgr = cv2.imread('data/assignment1/77.jpg')
after_bgr = cv2.imread('data/77_enhanced.jpg')

# Convert BGR to RGB
before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)

# Compute metrics
print("=" * 60)
print("OBJECTIVE EVALUATION REPORT")
print("=" * 60)

# Sharpness (Laplacian Variance)
sharp_before = compute_laplacian_variance(before_rgb)
sharp_after = compute_laplacian_variance(after_rgb)
print("\nSharpness (Variance of Laplacian):")
print(f"  Before: {sharp_before:.6f}")
print(f"  After:  {sharp_after:.6f}")
change_sharp = ((sharp_after - sharp_before) / sharp_before * 100) if sharp_before != 0 else 0
print(f"  Change: {change_sharp:+.2f}% (↑ higher is better)")

# Global Contrast (Std Dev of Luminance)
contrast_before = compute_global_contrast(before_rgb)
contrast_after = compute_global_contrast(after_rgb)
print("\nGlobal Contrast (Std Dev of Luminance):")
print(f"  Before: {contrast_before:.6f}")
print(f"  After:  {contrast_after:.6f}")
change_contrast = ((contrast_after - contrast_before) / contrast_before * 100) if contrast_before != 0 else 0
print(f"  Change: {change_contrast:+.2f}% (↑ higher is better)")

# Shadow Visibility (Fraction below 20)
shadow_before = compute_shadow_visibility(before_rgb, threshold=20)
shadow_after = compute_shadow_visibility(after_rgb, threshold=20)
print("\nShadow Visibility (Fraction < 20):")
print(f"  Before: {shadow_before:.6f} ({shadow_before*100:.2f}%)")
print(f"  After:  {shadow_after:.6f} ({shadow_after*100:.2f}%)")
change_shadow = ((shadow_before - shadow_after) / shadow_before * 100) if shadow_before != 0 else 0
print(f"  Change: {change_shadow:+.2f}% (↓ lower is better)")

# Overexposure Fraction (Fraction above 235)
over_before = compute_overexposure_fraction(before_rgb, threshold=235)
over_after = compute_overexposure_fraction(after_rgb, threshold=235)
print("\nOverexposure Fraction (Fraction > 235):")
print(f"  Before: {over_before:.6f} ({over_before*100:.2f}%)")
print(f"  After:  {over_after:.6f} ({over_after*100:.2f}%)")
change_over = ((over_before - over_after) / over_before * 100) if over_before != 0 else 0
print(f"  Change: {change_over:+.2f}% (↓ lower is better)")

print("\n" + "=" * 60)


# %% md
## 5.3 Visualization

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Objective Evaluation Metrics: Before vs After', fontsize=14, fontweight='bold')

# Sharpness comparison
ax = axes[0, 0]
categories = ['Before', 'After']
values = [sharp_before, sharp_after]
colors = ['#ff6b6b', '#51cf66']
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{value:.2f}',
           ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Variance', fontweight='bold')
ax.set_title('Sharpness (Laplacian Variance)', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Contrast comparison
ax = axes[0, 1]
values = [contrast_before, contrast_after]
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{value:.2f}',
           ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Std Deviation', fontweight='bold')
ax.set_title('Global Contrast (Luminance Std Dev)', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Shadow visibility comparison
ax = axes[1, 0]
values = [shadow_before * 100, shadow_after * 100]
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{value:.2f}%',
           ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Shadow Visibility (Pixels < 20)', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Overexposure comparison
ax = axes[1, 1]
values = [over_before * 100, over_after * 100]
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{value:.2f}%',
           ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Overexposure (Pixels > 235)', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('data/task5_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# %% md
## 5.4 Analysis & Discussion

# Create summary table
import pandas as pd

summary_data = {
    'Metric': [
        'Sharpness (Var Laplacian)',
        'Global Contrast (Std Dev)',
        'Shadow Visibility (%)',
        'Overexposure (%)'
    ],
    'Before': [
        f'{sharp_before:.6f}',
        f'{contrast_before:.6f}',
        f'{shadow_before*100:.2f}%',
        f'{over_before*100:.2f}%'
    ],
    'After': [
        f'{sharp_after:.6f}',
        f'{contrast_after:.6f}',
        f'{shadow_after*100:.2f}%',
        f'{over_after*100:.2f}%'
    ],
    'Change': [
        f'{change_sharp:+.2f}%',
        f'{change_contrast:+.2f}%',
        f'{change_shadow:+.2f}%',
        f'{change_over:+.2f}%'
    ]
}

df_summary = pd.DataFrame(summary_data)
print("\nSummary Table:")
print(df_summary.to_string(index=False))

# %% md
## 5.5 Questions & Answers (for Report)

print("""
QUESTIONS & DISCUSSION POINTS:

1. Do the metrics agree with perceived improvement? Explain any mismatch.
   
   Analysis:
   - Sharpness (Laplacian Variance): Shows the degree of high-frequency content.
     A metric increase indicates enhanced edges and details.
   
   - Global Contrast: Higher std dev indicates better tonal separation.
     Important for night enhancement where details in shadows are critical.
   
   - Shadow Visibility: The fraction of very dark pixels (< 20).
     For night enhancement, this should DECREASE significantly.
   
   - Overexposure: Ensures we don't blow out bright areas while lifting shadows.

2. Which metric is most informative for night enhancement?
   
   For night image enhancement, SHADOW VISIBILITY is typically most informative
   because:
   - It directly measures the primary goal: recovering detail in dark regions
   - It's intuitive and correlates well with perceived improvement
   - It helps balance the trade-off between shadow lifting and not blowing out highlights
   
   Secondary metric: Global Contrast is valuable as it shows whether details are
   separable (higher contrast = better visibility) while also preventing clipping.
""")

