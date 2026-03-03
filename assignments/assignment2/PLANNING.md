## Plan: Build Complete Historical Photo Restoration Pipeline

Implement an end-to-end restoration pipeline that removes synthetic degradations (fading, grain, dust, scratches) from historical photos. The pipeline will apply filters in sequence (Gaussian smoothing, median filtering, morphological operations, inpainting), validate results with quantitative metrics (high-frequency residual, edge density, SSIM/PSNR), and demonstrate effectiveness on multiple degraded images.

### Steps

1. **Enhance `ImageDegrader` class** – Add `compute_dft_magnitude()`, fix parameter passing, and debug frequency analysis functions.

2. **Implement individual restoration filters** – Create methods for Gaussian smoothing, median filtering, Sobel gradients, and morphological operations with configurable parameters.

3. **Build defect mask generation** – Implement three masking strategies: gradient magnitude thresholding, intensity thresholding, and morphological top/black-hat transforms.

4. **Construct restoration pipeline** – Chain filters in optimal order (smoothing → median → morphology → inpainting) with documented parameter justification.

5. **Add quantitative metrics** – Implement high-frequency residual (grain), edge density, SSIM, and PSNR calculation functions.

6. **Validate on test images** – Run pipeline on degraded photos, capture intermediate outputs, compare with original, and document results.

### Further Considerations

1. **Parameter tuning**: What Gaussian σ values to test (e.g., 1.0, 2.0, 5.0)? What median kernel sizes (3, 5, 7)? What morphological kernel shapes/sizes?

2. **Filter ordering**: Should median filtering precede or follow Gaussian smoothing? Should morphological operations be applied before or after inpainting?

3. **Metric selection**: Will you use high-frequency residual std-dev, edge density ratios, or SSIM/PSNR? How will you weight these in final evaluation?

---

## Detailed Code Architecture

### Module 1: Degradation & Utilities (`ImageDegrader` class)

**Location**: Implement in notebook or separate `restoration_utils.py`

**Methods**:
- `__init__(img_path)` – Initialize with image path and RNG
  - `self.img_path = img_path`
  - `self.rng = np.random.default_rng()` for reproducibility
  
- `add_fading(img, strength=0.35)` → degraded_img
  - Reduce contrast: `img * (1 - strength) + 128 * strength`
  - Apply vignette using radial gradient
  
- `add_grain(img, sigma=9.0)` → degraded_img
  - Generate Gaussian noise: `self.rng.normal(0, sigma, img.shape)`
  - Add to image and clip to [0, 255]
  
- `add_dust(img, p=0.006)` → degraded_img
  - Create random binary mask with probability p
  - Replace masked pixels with random 0 or 255
  
- `add_scratches(img, n_lines=11)` → degraded_img
  - For each line: random start/end points, thickness, color (210–255)
  - Use `cv.line()` with `cv.LINE_AA` for anti-aliasing
  
- `degrade(img, **params)` → degraded_img
  - Apply all four effects sequentially
  - Return combined degraded image
  
- `save_incremented(img, directory="old_Photos")` → output_path
  - Auto-increment filename: `old_photo_1.jpg`, `old_photo_2.jpg`, etc.
  - Create directory if needed

### Module 2: Frequency & Gradient Analysis

**Methods**:
- `compute_dft_magnitude(img)` → (mag_u8, Fshift)
  - Compute 2D FFT: `F = np.fft.fft2(img.astype(np.float32))`
  - Shift zero frequency to center: `Fshift = np.fft.fftshift(F)`
  - Log-magnitude: `mag = np.log(np.abs(Fshift) + 1)`
  - Normalize to 0–255: `mag_u8 = ((mag - min) / (max - min) * 255).astype(uint8)`
  - Return both for visualization and analysis
  
- `compute_gradient(img, ksize=3)` → (Gx, Gy, magnitude)
  - X-derivative: `Gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize)`
  - Y-derivative: `Gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize)`
  - Magnitude: `mag = np.sqrt(Gx² + Gy²)`
  - Return all three; allow ksize ∈ {1, 3, 5, 7}
  
- `compute_morphological_gradient(img, kernel)` → morph_grad
  - Dilation: `dil = cv.dilate(img, kernel, iterations=1)`
  - Erosion: `ero = cv.erode(img, kernel, iterations=1)`
  - Gradient: `morph_grad = cv.subtract(dil, ero)`
  
- `visualize_spectrum(mag_u8, title="DFT Magnitude", save_path=None)`
  - Display with `plt.imshow(mag_u8, cmap='gray')`
  - Add colorbar and title
  - Save to `figs/` if path provided

### Module 3: Restoration Filters

**Methods**:
- `gaussian_smooth(img, sigma)` → smoothed_img
  - `ksize` computed from sigma (e.g., `ksize = int(6*sigma) | 1`)
  - Use `cv.GaussianBlur(img, (ksize, ksize), sigma)`
  
- `median_filter(img, ksize=5)` → filtered_img
  - `ksize` must be odd and ≥ 3
  - Use `cv.medianBlur(img, ksize)`
  
- `create_defect_mask_gradient(img, threshold_pct=0.35)` → binary_mask
  - Compute gradient magnitude: `gmag = compute_gradient(img)[2]`
  - Threshold: `thr = threshold_pct * gmag.max()`
  - Binary mask: `mask[gmag > thr] = 255`
  
- `create_defect_mask_intensity(img, threshold_pct=0.35)` → binary_mask
  - Threshold on pixel intensity: `thr = threshold_pct * 255`
  - Mark bright outliers: `mask = (img > thr) | (img < 255 - thr)`
  
- `create_defect_mask_tophat(img, kernel)` → binary_mask
  - Opening: `opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)`
  - Top-hat: `tophat = cv.subtract(img, opened)`
  - Threshold: `mask = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)`
  
- `refine_mask(mask, kernel, operations=['open', 'close'])` → refined_mask
  - Apply opening: removes small noise
  - Apply closing: fills small holes
  - Return cleaned binary mask
  
- `inpaint(img, mask, radius=3)` → inpainted_img
  - Use `cv.inpaint(img, mask, radius, cv.INPAINT_TELEA)`
  - Telea method: fast, good for thin scratches

### Module 4: Evaluation Metrics

**Methods**:
- `high_freq_residual(img)` → float
  - Compute high-frequency components via Laplacian or high-pass filter
  - Return std-dev of high-freq energy (proxy for grain)
  
- `edge_density(img, threshold)` → float
  - Compute edges: `edges = cv.Canny(img, threshold, 2*threshold)`
  - Density = (edge_pixels / total_pixels) * 100
  
- `compute_ssim(img1, img2)` → float
  - Use `skimage.metrics.structural_similarity()`
  - Return SSIM value in [0, 1]
  
- `compute_psnr(img1, img2)` → float
  - MSE = mean((img1 - img2)²)
  - PSNR = 20 * log10(255 / sqrt(MSE))
  
- `compare_metrics(original, degraded, restored)` → metrics_dict
  - High-freq residual: [degraded, restored]
  - Edge density: [original, restored]
  - SSIM vs original: [degraded, restored]
  - PSNR vs original: [degraded, restored]
  - Return dict with all values

### Restoration Pipeline Function

```python
def restore_image(degraded_img, params_dict):
    """
    Params dictionary keys:
      - gaussian_sigma (float): smoothing strength
      - median_ksize (int, odd): dust removal kernel
      - grad_threshold_pct (float 0–1): defect detection sensitivity
      - morph_kernel_shape (str): 'RECT', 'ELLIPSE', or 'CROSS'
      - morph_kernel_size (int, odd): 3, 5, 7, etc.
      - inpaint_radius (int): neighborhood for inpainting
      - use_clahe (bool): apply CLAHE before pipeline
      - final_enhance (bool): apply histogram equalization after
    """
    out = degraded_img.copy()
    
    # Step 1: Optional contrast normalization (CLAHE)
    if params_dict.get('use_clahe', False):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
    
    # Step 2: Gaussian smoothing (grain reduction)
    sigma = params_dict.get('gaussian_sigma', 2.0)
    out = gaussian_smooth(out, sigma)
    
    # Step 3: Median filtering (dust removal)
    ksize = params_dict.get('median_ksize', 5)
    out = median_filter(out, ksize)
    
    # Step 4: Defect mask generation
    grad_mag = compute_gradient(out, ksize=3)[2]
    mask = create_defect_mask_gradient(
        grad_mag, 
        params_dict.get('grad_threshold_pct', 0.35)
    )
    
    # Step 5: Morphological refinement
    kernel_shape = params_dict.get('morph_kernel_shape', 'ELLIPSE')
    kernel_size = params_dict.get('morph_kernel_size', 5)
    kernel = cv.getStructuringElement(
        getattr(cv, f"MORPH_{kernel_shape}"),
        (kernel_size, kernel_size)
    )
    mask = refine_mask(mask, kernel, ['open', 'close'])
    
    # Step 6: Inpainting (scratch removal)
    radius = params_dict.get('inpaint_radius', 3)
    out = inpaint(out, mask, radius)
    
    # Step 7: Final contrast enhancement
    if params_dict.get('final_enhance', False):
        out = cv.equalizeHist(out)
    
    return out
```

---

## Parameter Study Design

### Phase 1: Individual Filter Tuning

**Gaussian Smoothing**
- Test σ ∈ {1.0, 2.0, 3.0, 5.0}
- Measure: high-frequency residual, edge retention
- Goal: Find σ that removes grain without blurring edges

**Median Filtering**
- Test k ∈ {3, 5, 7}
- Measure: dust removal (low values in mask), edge preservation
- Goal: Find k that removes salt-and-pepper without over-smoothing

**Gradient Threshold**
- Test threshold_pct ∈ {0.25, 0.35, 0.40}
- Measure: defect detection accuracy, false positives
- Goal: Balance between removing defects and preserving true edges

### Phase 2: Morphological Refinement

**Kernel Shapes & Sizes**
- Shapes: RECT vs. ELLIPSE (CROSS less common)
- Sizes: 3×3, 5×5, 7×7
- Operations: open-then-close vs. close-then-open
- Measure: mask connectivity, false positives/negatives

### Phase 3: End-to-End Validation

- Apply full pipeline with best parameters on ≥3 test images
- Capture intermediate outputs at each stage
- Compute metrics at each stage; build comparison table
- Document visual quality and quantitative improvements

---

## Notebook Workflow

1. **Cell 1: Imports & Setup**
   - Import cv2, numpy, matplotlib, skimage.metrics
   - Set random seed for reproducibility
   - Create `figs/` directory if needed

2. **Cell 2: Load Original Image**
   - Read grayscale historical photo from `Old_Photos/`
   - Display original

3. **Cell 3: Degrade Image**
   - Instantiate `ImageDegrader`
   - Apply degradation with standard parameters
   - Save degraded image

4. **Cell 4: Frequency Analysis (5 pts)**
   - Compute and display DFT magnitude spectrum
   - Interpret: low frequencies (center), high frequencies (grain)

5. **Cell 5: Gaussian Smoothing Study (10 pts)**
   - Test σ ∈ {1.0, 2.0, 3.0, 5.0}
   - Side-by-side comparison; visualize
   - Justify chosen σ

6. **Cell 6: Median vs. Gaussian (5 pts)**
   - Compare dust removal and edge preservation
   - Document which is better for impulse noise

7. **Cell 7: Gradient & Morphology (10 pts)**
   - Compute Sobel gradients on original and smoothed
   - Test Sobel kernel sizes {1, 3, 5, 7}
   - Compare Sobel vs. morphological gradient

8. **Cell 8: Defect Mask Generation (5 pts)**
   - Create masks via: gradient threshold, intensity threshold, top-hat
   - Display all three; justify choice

9. **Cell 9: Morphological Refinement (5 pts)**
   - Apply open/close with different kernel sizes/shapes
   - Show refined mask quality

10. **Cell 10: Inpainting (implicit in pipeline)**
    - Show before/after inpainting on mask

11. **Cell 11: Pipeline Design & Justification (25 pts)**
    - Define `restore_image()` with chosen parameters
    - Document ordering and parameter rationale in text

12. **Cell 12: Quantitative Metrics (implicit)**
    - Implement metric functions
    - Display as table: [metric, degraded, restored]

13. **Cell 13: Evidence (25 pts)**
    - Run pipeline on test image
    - Save intermediate outputs: degraded → smoothed → median → mask → inpainted → final
    - Display in grid
    - Compute metrics; create summary table



## File Output Organization

```
---figs/
  ├── original_01/  (first test image and all its analysis)
  ├── original_02/  (second test image and all its analysis)
  └── original_03/  (third test image and all its analysis)
  
```

```
figs/
  ├── original_01/
  │   ├── original.png
  │   ├── degraded.png
  │   ├── dft_spectrum.png
  │   ├── gaussian_study/
  │   │   ├── gaussian_sigma_1.0.png
  │   │   ├── gaussian_sigma_2.0.png
  │   │   ├── gaussian_sigma_3.0.png
  │   │   └── gaussian_sigma_5.0.png
  │   ├── gradient_study/
  │   │   ├── gradient_sobel_k1.png
  │   │   ├── gradient_sobel_k3.png
  │   │   ├── gradient_sobel_k5.png
  │   │   └── gradient_morph.png
  │   ├── mask_generation/
  │   │   ├── mask_gradient.png
  │   │   ├── mask_intensity.png
  │   │   └── mask_tophat.png
  │   ├── pipeline/
  │   │   ├── step1_original.png
  │   │   ├── step2_gaussian.png
  │   │   ├── step3_median.png
  │   │   ├── step4_mask.png
  │   │   ├── step5_morph_refined.png
  │   │   ├── step6_inpainted.png
  │   │   └── step7_final.png
  │   └── metrics_summary.png
  ├── original_02/
  │   ├── original.png
  │   ├── degraded.png
  │   ├── dft_spectrum.png
  │   ├── gaussian_study/
  │   │   ├── gaussian_sigma_1.0.png
  │   │   ├── gaussian_sigma_2.0.png
  │   │   ├── gaussian_sigma_3.0.png
  │   │   └── gaussian_sigma_5.0.png
  │   ├── gradient_study/
  │   │   ├── gradient_sobel_k1.png
  │   │   ├── gradient_sobel_k3.png
  │   │   ├── gradient_sobel_k5.png
  │   │   └── gradient_morph.png
  │   ├── mask_generation/
  │   │   ├── mask_gradient.png
  │   │   ├── mask_intensity.png
  │   │   └── mask_tophat.png
  │   ├── pipeline/
  │   │   ├── step1_original.png
  │   │   ├── step2_gaussian.png
  │   │   ├── step3_median.png
  │   │   ├── step4_mask.png
  │   │   ├── step5_morph_refined.png
  │   │   ├── step6_inpainted.png
  │   │   └── step7_final.png
  │   └── metrics_summary.png
  └── original_03/
      ├── original.png
      ├── degraded.png
      ├── dft_spectrum.png
      ├── gaussian_study/
      │   ├── gaussian_sigma_1.0.png
      │   ├── gaussian_sigma_2.0.png
      │   ├── gaussian_sigma_3.0.png
      │   └── gaussian_sigma_5.0.png
      ├── gradient_study/
      │   ├── gradient_sobel_k1.png
      │   ├── gradient_sobel_k3.png
      │   ├── gradient_sobel_k5.png
      │   └── gradient_morph.png
      ├── mask_generation/
      │   ├── mask_gradient.png
      │   ├── mask_intensity.png
      │   └── mask_tophat.png
      ├── pipeline/
      │   ├── step1_original.png
      │   ├── step2_gaussian.png
      │   ├── step3_median.png
      │   ├── step4_mask.png
      │   ├── step5_morph_refined.png
      │   ├── step6_inpainted.png
      │   └── step7_final.png
      └── metrics_summary.png
```

**Note**: Each test image gets its own subdirectory under `figs/` (e.g., `original_01/`, `original_02/`, `original_03/`) containing all analysis outputs for that image. This keeps results organized when testing the pipeline on multiple historical photos.

---

## Key Design Justifications

### 1. Filter Ordering: Gaussian → Median → Morphology → Inpaint

- **Gaussian first**: Smooths diffuse grain noise; enables cleaner gradient computation
- **Median second**: Removes salt-and-pepper dust non-linearly; preserves edges better than Gaussian alone
- **Morphology third**: Cleans structured scratches via erosion/dilation; removes small noise in mask
- **Inpaint last**: Fills remaining holes in known defect regions; avoids over-processing

### 2. Gaussian vs. Median Trade-off

- **Gaussian**: Linear, invertible, stable, but blurs edges
- **Median**: Nonlinear, breaks edges near boundaries, but preserves discontinuities
- **Combined**: Gaussian handles grain (distributed noise), median handles dust (impulse noise)

### 3. Defect Mask Strategy: Gradient-Based + Morphology

- **Gradient threshold**: Detects all strong edges (true scene edges + scratches)
- **Morphological refinement**: Open removes thin isolated noise; close fills small gaps
- **Why not just intensity?** Scratches are bright (high intensity), but so are legitimate scene features; gradient distinguishes *thin* structures

### 4. Metrics Justification

- **High-freq residual**: Directly measures grain (noise) remaining; proxy for quality
- **Edge density**: Ensures edges are preserved (comparison original vs. restored)
- **SSIM**: Perceptual similarity; accounts for luminance, contrast, structure
- **PSNR**: Standard metric; sensitive to large errors but easy to interpret

---

## Success Criteria

✓ All six learning objectives addressed  
✓ Parameter choices justified quantitatively  
✓ Visual evidence (≥3 intermediate outputs per test image)  
✓ ≥2 quantitative metrics with tables/charts  
✓ Pipeline documented with pseudocode and rationale  
✓ Report includes interpretation of results  
