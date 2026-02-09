# Task 2 Questions

## 1. Which method gives the best perceptual improvement before denoising? Why?

**Gamma + CLAHE combination** typically gives the best perceptual improvement for night images because:

- **Gamma correction alone** (γ < 1) lifts shadows globally but can wash out already-bright regions and doesn't improve local contrast
- **CLAHE alone** enhances local contrast but doesn't address the overall darkness of underexposed images
- **Combined approach**: Gamma first lifts the dark tones into a visible range, then CLAHE enhances local details and contrast adaptively. This two-step process:
  1. Makes shadow details visible (gamma)
  2. Improves local contrast without over-amplifying already-bright areas (CLAHE's clip limiting)

The percentile stretch is simpler but less adaptive—it only remaps the global histogram without considering local variations.

---

## 2. Explain how clipLimit and tileGridSize influence CLAHE artifacts

| Parameter | Effect | Artifacts |
|-----------|--------|-----------|
| **clipLimit** | Controls maximum contrast amplification per tile | **Low values (1-2)**: Subtle enhancement, fewer artifacts. **High values (>4)**: Over-enhancement, noise amplification, "halo" effects around edges |
| **tileGridSize** | Determines the size of local regions for histogram equalization | **Small tiles (4×4)**: More localized adaptation but visible block boundaries and checkerboard patterns. **Large tiles (16×16)**: Smoother results but less localized—approaches global histogram equalization |

**Key tradeoffs:**
- High `clipLimit` + small `tileGridSize` = maximum artifacts (blocky, noisy)
- Low `clipLimit` + large `tileGridSize` = minimal artifacts but weaker enhancement
- Typical balanced settings: `clipLimit=2.0-3.0`, `tileGridSize=(8,8)`

---

## 3. Why is γ < 1 typically used for night enhancement?

The gamma correction formula is:

$$L' = 255 \left(\frac{L}{255}\right)^\gamma$$

When **γ < 1**:

- **Dark pixels are lifted more than bright pixels** — the curve is concave, providing stronger amplification in shadows
- For example, with γ = 0.5:
  - A pixel at intensity 25 → 255 × (25/255)^0.5 ≈ **80** (3× boost)
  - A pixel at intensity 200 → 255 × (200/255)^0.5 ≈ **226** (only 1.1× boost)

This **non-linear mapping** is ideal for night images because:

1. Most information is compressed in the dark end of the histogram
2. Bright areas (lights, highlights) don't need additional boosting
3. It mimics human visual perception, which is more sensitive to relative changes in dark regions

Using γ > 1 would darken the image further—the opposite of what's needed.

The user wants to move the Task 2 questions and answers to a markdown cell in the notebook. I'll add this content to a new markdown cell in the notebook.

