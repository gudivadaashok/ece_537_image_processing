# Task 5: Objective Evaluation - Complete Guide

## Overview

Task 5 requires computing quantitative metrics to evaluate the quality of image enhancement. Three primary metrics are essential:

1. **Sharpness Proxy (Variance of Laplacian)**
2. **Global Contrast (Standard Deviation of Luminance)**
3. **Shadow Visibility (Fraction of Pixels Below Threshold)**

An additional metric (Overexposure Fraction) is useful for ensuring we don't blow out bright areas.

---

## Metric Definitions & Interpretation

### 1. Sharpness Proxy: Variance of Laplacian

**Mathematical Definition:**
```
S = Var(∇²I)
```

Where:
- ∇² is the Laplacian operator (second-order spatial derivative)
- Var() computes the variance of the result
- I is the image (or luminance channel)

**Interpretation:**
- **High variance** → Sharp edges, high-frequency content preserved
- **Low variance** → Blurry, low-frequency dominant
- **For night enhancement:** Should increase significantly after sharpening (Task 4)
- **Trade-off:** Over-sharpening can amplify noise

**Why this metric?**
- Quantifies edge acuity without human judgment
- Widely used in autofocus systems and image quality assessment
- Correlates well with perceived sharpness in well-denoised images

---

### 2. Global Contrast: Standard Deviation of Luminance

**Mathematical Definition:**
```
C = σ(L) = √(E[L² - (E[L])²])
```

Where:
- L is the luminance (gray channel)
- σ is standard deviation
- E[] is the expectation (mean)

**Interpretation:**
- **High std dev** → Colors/tones well-separated, good visibility
- **Low std dev** → Flat histogram, poor tonal separation
- **For night enhancement:** Should increase after tone/contrast adjustments
- **Relationship with shadow lifting:** Gamma correction and CLAHE increase this

**Why this metric?**
- Direct measure of brightness range utilization
- Indicates whether image details are visually separable
- Sensitive to both shadow lifting and overall tonal balance

---

### 3. Shadow Visibility: Fraction of Pixels Below Threshold

**Mathematical Definition:**
```
V = |{i,j : L[i,j] < T}| / (M × N)
```

Where:
- T is the shadow threshold (commonly 20)
- L[i,j] is the luminance at pixel (i,j)
- M × N is the total number of pixels
- |·| is cardinality (count)

**Interpretation:**
- **Value = 0.55** → 55% of pixels are "deep shadow" (< value of 20)
- **Value = 0.35** → 35% of pixels are "deep shadow" (improved)
- **For night enhancement:** Should DECREASE significantly
- **Perfect night enhancement:** Moves shadow pixels out of the <20 range into the 20-100+ range

**Why this metric?**
- Directly measures the primary goal: recovering shadow detail
- Threshold of 20 is typical (represents near-black pixels)
- Intuitive interpretation: "How much of the image was still very dark?"
- Most critical metric for night image assessment

---

### 4. Overexposure Fraction (Bonus Metric)

**Mathematical Definition:**
```
O = |{i,j : L[i,j] > T}| / (M × N)
```

Where:
- T is the overexposure threshold (commonly 235, near-white)

**Interpretation:**
- **Low fraction** → Few blown-out pixels (good)
- **High fraction** → Many saturated/clipped bright areas (bad)
- **Typical values:** < 1% is reasonable
- **Watch for:** Sharp increases in overexposure when aggressively lifting shadows

---

## Sample Results Interpretation

Based on the actual execution above:

```
Before Enhancement:
  Sharpness:         70.64
  Contrast:          36.40
  Shadow Fraction:   55.47% (0.5547)
  Overexposure:      0.048%

After Enhancement:
  Sharpness:         313.61  (+343.96%)
  Contrast:          52.47   (+44.16%)
  Shadow Fraction:   34.66%  (-37.52% improvement)
  Overexposure:      0.38%   (+783% - CAUTION: increased)
```

### Analysis:

✅ **Sharpness improved dramatically** (+344%)
  - Indicates successful sharpening (unsharp mask)
  - Edges are now much more pronounced

✅ **Contrast improved significantly** (+44%)
  - Shows better tonal separation
  - Details in mid-tones are more visible

✅ **Shadows lifted substantially** (-37.5%)
  - Excellent shadow recovery
  - Deep shadows reduced from 55% to 35% of image

⚠️ **Overexposure increased** (slight)
  - Went from 0.048% to 0.38%
  - Still very minimal, but worth noting
  - Trade-off acceptable for shadow lifting benefits

---

## Questions for Your Report

### Question 1: Do the metrics agree with perceived improvement? Explain any mismatch.

**Answer Framework:**

Metrics typically correlate well with perceived quality in night enhancement when:

1. **Sharpness & Contrast Both Increase:**
   - Indicates details are both enhanced AND visible
   - Good sign of successful pipeline

2. **Shadow Visibility Decreases Significantly:**
   - MUST happen for night enhancement
   - This is the primary goal

3. **Overexposure Remains Low:**
   - Ensures we don't sacrifice highlights
   - Indicates balanced enhancement

**Common Mismatches:**

- **High Sharpness but Low Contrast:** Over-sharpening in noisy regions; appears grainy
- **High Contrast but Low Sharpness:** Good tonal separation but blurry edges; poor detail visibility
- **Sharpness Increases, Shadows Don't Lift:** Wrong enhancement focus (e.g., gamma too weak)
- **All metrics improve, but image looks noisy:** Noise amplification; increase denoising strength

**From your results:**
All three primary metrics improved significantly and consistently, which suggests your enhancement pipeline is working well. The slight increase in overexposure is a minor concern but acceptable.

---

### Question 2: Which metric is most informative for night enhancement?

**Answer: Shadow Visibility**

**Reasoning:**

1. **Direct Relevance:**
   - Night enhancement's primary goal is to make dark areas visible
   - Shadow Visibility directly measures this goal
   - Tells you exactly how much shadow recovery occurred

2. **Interpretability:**
   - Simple percentage: "55% → 35% of pixels were lifted out of shadows"
   - Client/stakeholder friendly: "We reduced deep shadows by 38%"
   - No special knowledge required to understand

3. **Optimal Range:**
   - < 10% is excellent (most pixels visible)
   - 10-30% is good (majority of image visible)
   - > 50% is poor (half the image still very dark)

4. **Guides Pipeline Design:**
   - If shadow visibility is high, apply stronger gamma or CLAHE
   - If shadow visibility is low but sharpness is low too, denoise first
   - Helps identify bottleneck in your pipeline

**Secondary Important Metrics:**

- **Global Contrast:** Ensures lifted shadows have visible detail (high contrast = visible details)
- **Sharpness:** Ensures details don't get lost to blur or noise

**Less Critical for Night Work:**

- **Overexposure:** Secondary concern; mainly ensures we're not clipping highlights

**Combined Strategy:**

Best assessment uses **both** Shadow Visibility AND Global Contrast:
- Shadow Visibility: "Did we recover the dark regions?"
- Global Contrast: "Are the recovered details actually visible?"

For example:
- Shadow ↓, Contrast ↑ = **Excellent** ✓
- Shadow ↓, Contrast ↓ = **Poor** ✗ (recovered shadows but they're invisible)
- Shadow ↑, Contrast ↑ = **Wrong approach** (sharpening only, not lifting)

---

## Implementation in Jupyter Notebook

The complete implementation is provided in `Task5_Notebook_Cells.py`. Copy the cells in order:

1. **Section 5.1:** Define all metric functions
2. **Section 5.2:** Load images and compute metrics (prints report)
3. **Section 5.3:** Visualization (creates comparison plots)
4. **Section 5.4:** Summary table and discussion

The script will:
- Print a formatted evaluation report
- Create a 2×2 subplot comparison visualization
- Save `data/task5_metrics_comparison.png`

---

## Common Pitfalls & Solutions

### Pitfall 1: Image size mismatch
**Problem:** Before and after images are different sizes
**Solution:** Resize them to match before comparing
```python
if before_rgb.shape != after_rgb.shape:
    after_rgb = cv2.resize(after_rgb, (before_rgb.shape[1], before_rgb.shape[0]))
```

### Pitfall 2: Using BGR instead of RGB
**Problem:** Color spaces confused; metrics may be wrong
**Solution:** Always convert BGR→RGB with `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`

### Pitfall 3: Computing on wrong channel
**Problem:** Applying sharpness metric to RGB separately instead of grayscale
**Solution:** Always convert to grayscale first for these metrics
```python
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

### Pitfall 4: Wrong threshold for shadow visibility
**Problem:** Using threshold that's too high/low, making metric less meaningful
**Solution:** Use standard thresholds (20 for shadow, 235 for overexposure)

### Pitfall 5: Not handling edge cases
**Problem:** Division by zero if before_value = 0
**Solution:** Check before computing percent change
```python
change = ((after - before) / before * 100) if before != 0 else float('inf')
```

---

## Verification Checklist

Before submitting, verify:

- [ ] All three primary metrics computed and reported
- [ ] Before/After images loaded correctly
- [ ] Results printed with clear labels
- [ ] Visualization saved as PNG file
- [ ] Shadow visibility shows DECREASE (improvement)
- [ ] Contrast shows INCREASE (improvement)
- [ ] Sharpness increase correlates with denoising effectiveness
- [ ] Overexposure fraction remains low (< 2%)
- [ ] Report answers both questions with detailed analysis
- [ ] Metrics values have appropriate precision (4-6 decimal places)

---

## Reference Code Snippets

### Compute all metrics at once:
```python
metrics = {
    'Sharpness_before': compute_laplacian_variance(before_rgb),
    'Sharpness_after': compute_laplacian_variance(after_rgb),
    'Contrast_before': compute_global_contrast(before_rgb),
    'Contrast_after': compute_global_contrast(after_rgb),
    'Shadow_before': compute_shadow_visibility(before_rgb, threshold=20),
    'Shadow_after': compute_shadow_visibility(after_rgb, threshold=20),
}
```

### Create percent change dict:
```python
changes = {
    'Sharpness': ((metrics['Sharpness_after'] - metrics['Sharpness_before']) 
                  / metrics['Sharpness_before'] * 100),
    'Contrast': ((metrics['Contrast_after'] - metrics['Contrast_before']) 
                 / metrics['Contrast_before'] * 100),
    'Shadow': ((metrics['Shadow_before'] - metrics['Shadow_after']) 
               / metrics['Shadow_before'] * 100),  # Note: inverted for shadow!
}
```

---

## Additional Resources

- **OpenCV Laplacian:** `cv2.Laplacian()`
- **NumPy statistics:** `np.var()`, `np.std()`
- **Image quality metrics:** Consult "Image Quality Assessment: From Error Visibility to Structural Similarity"

---

**Good luck with your enhancement pipeline!** 🚀

