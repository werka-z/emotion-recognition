# Facial Landmark Detection Coverage — FER-2013

Tested whether **dlib** and **mediapipe** can reliably find faces/landmarks on FER-2013's
48×48 grayscale crops, on a stratified random sample of 999 images from `data/train`
(seed=42, proportional to class size). Script: [detect_landmarks.py](detect_landmarks.py).

## Results

| Method | Detected | Failed | Time |
|---|---|---|---|
| dlib, native 48×48 | 686/999 (68.7%) | 313 (31.3%) | 0.6 ms/img |
| dlib, 4x upscaled (cubic) | 708/999 (70.9%) | 291 (29.1%) | 6.4 ms/img |
| mediapipe, native 48×48 | 888/999 (88.9%) | 111 (11.1%) | 4.1 ms/img |

## Findings

- **dlib's HOG-based detector struggles at native FER-2013 resolution**, missing ~31% of faces.
- **Upscaling barely helps dlib** (+2.2 pts for 10x the runtime) — confirms the misses aren't
  primarily a resolution problem, but bad crops, occlusion, extreme pose, or non-frontal faces.
  Manually checked (`contact_sheet_dlib_fixed_by_upscale.png`): the images "fixed" by upscaling
  are genuinely borderline/ambiguous faces, not clean detections unlocked by more pixels.
- **mediapipe is far more robust** (88.9% vs 68.7%) and still fast.
- **86 images (8.6%) fail under every method** (dlib@1x, dlib@4x, mediapipe) — manually
  inspected via `contact_sheet_ALL_METHODS_failed.png` and confirmed to be **genuinely bad
  crops** (non-faces, heavy occlusion, cropped-off faces), not detector bugs.

## Recommendation

Use mediapipe as the primary landmark extractor for this dataset; treat dlib failures that
mediapipe also misses as the manual-review/drop candidates, not the full dlib failure set.

## Files

| File | Contents |
|---|---|
| `results/failures_dlib_upscale1x.txt` | Images where dlib found no face at native res |
| `results/failures_dlib_upscale4x.txt` | Images where dlib found no face after 4x upscale |
| `results/failures_mediapipe_native.txt` | Images where mediapipe found no face |
| `results/failures_ALL_METHODS.txt` | Images that failed under all three settings (last-resort/manual-review set) |
| `results/fixed_by_upscale_list.txt` | Images dlib missed at 1x but caught at 4x |
| `results/contact_sheet_ALL_METHODS_failed.png` | Visual grid of the 86 all-methods failures |
| `results/contact_sheet_dlib_fixed_by_upscale.png` | Original vs. 4x-upscaled pairs for the 63 dlib upscale "fixes" |
| `results/contact_sheet_mediapipe_failed.png` | Visual grid of the 111 mediapipe failures |

All paths in the `.txt` files are relative to the `emotion-recognition` project root.
