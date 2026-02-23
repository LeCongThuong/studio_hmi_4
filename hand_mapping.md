# MANO-to-MHR70 Hand Keypoint Mapping

This file maps **MANO 21 hand keypoints** to the hand keypoints used in **`mhr70`**.

It is intended for pipelines where you infer hand keypoints from a hand-specific model (e.g., WiLoR / HaMeR) and inject them into MHR keypoints for downstream optimization.

## 1) Assumed MANO Ordering (WiLoR / HaMeR output)

WiLoR and HaMeR reorder MANO joints to an OpenPose-like hand order using:

```python
mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
```

So the **expected 21-joint order** for this mapping is:

- `0`: wrist
- `1..4`: thumb1, thumb2, thumb3, thumb4(tip)
- `5..8`: index1, index2, index3, index4(tip)
- `9..12`: middle1, middle2, middle3, middle4(tip)
- `13..16`: ring1, ring2, ring3, ring4(tip)
- `17..20`: pinky1, pinky2, pinky3, pinky4(tip)

If your model gives **raw MANO joint order** (not OpenPose-reordered), first apply the reorder above.

## 2) MHR70 Hand Indices

From `mhr70.py`:

- Right hand in full MHR70: indices `21..41` (including wrist at `41`)
- Left hand in full MHR70: indices `42..62` (including wrist at `62`)

Important naming note in `mhr70`:

- `*_third_joint` is the finger joint closest to wrist (base side)
- `*2`, `*3`, `*4` go toward fingertip (`*4` is tip)

## 3) MANO(21) -> MHR70 Index Mapping

| MANO idx | MANO name | Right MHR name | Right MHR idx | Left MHR name | Left MHR idx |
|---:|---|---|---:|---|---:|
| 0 | wrist | right_wrist | 41 | left_wrist | 62 |
| 1 | thumb1 | right_thumb_third_joint | 24 | left_thumb_third_joint | 45 |
| 2 | thumb2 | right_thumb2 | 23 | left_thumb2 | 44 |
| 3 | thumb3 | right_thumb3 | 22 | left_thumb3 | 43 |
| 4 | thumb4 (tip) | right_thumb4 | 21 | left_thumb4 | 42 |
| 5 | index1 | right_forefinger_third_joint | 28 | left_forefinger_third_joint | 49 |
| 6 | index2 | right_forefinger2 | 27 | left_forefinger2 | 48 |
| 7 | index3 | right_forefinger3 | 26 | left_forefinger3 | 47 |
| 8 | index4 (tip) | right_forefinger4 | 25 | left_forefinger4 | 46 |
| 9 | middle1 | right_middle_finger_third_joint | 32 | left_middle_finger_third_joint | 53 |
| 10 | middle2 | right_middle_finger2 | 31 | left_middle_finger2 | 52 |
| 11 | middle3 | right_middle_finger3 | 30 | left_middle_finger3 | 51 |
| 12 | middle4 (tip) | right_middle_finger4 | 29 | left_middle_finger4 | 50 |
| 13 | ring1 | right_ring_finger_third_joint | 36 | left_ring_finger_third_joint | 57 |
| 14 | ring2 | right_ring_finger2 | 35 | left_ring_finger2 | 56 |
| 15 | ring3 | right_ring_finger3 | 34 | left_ring_finger3 | 55 |
| 16 | ring4 (tip) | right_ring_finger4 | 33 | left_ring_finger4 | 54 |
| 17 | pinky1 | right_pinky_finger_third_joint | 40 | left_pinky_finger_third_joint | 61 |
| 18 | pinky2 | right_pinky_finger2 | 39 | left_pinky_finger2 | 60 |
| 19 | pinky3 | right_pinky_finger3 | 38 | left_pinky_finger3 | 59 |
| 20 | pinky4 (tip) | right_pinky_finger4 | 37 | left_pinky_finger4 | 58 |

## 4) Ready-to-Use Python Mapping

```python
# index = MANO/OpenPose hand index (0..20), value = MHR70 index (0..69)
MANO_TO_MHR70_RIGHT = [
    41, 24, 23, 22, 21,
    28, 27, 26, 25,
    32, 31, 30, 29,
    36, 35, 34, 33,
    40, 39, 38, 37,
]

MANO_TO_MHR70_LEFT = [
    62, 45, 44, 43, 42,
    49, 48, 47, 46,
    53, 52, 51, 50,
    57, 56, 55, 54,
    61, 60, 59, 58,
]

# Optional: if you need MHR hand-subset order (20 points, no wrist)
# mhr70.py right_hand_keypoint_names / left_hand_keypoint_names order
MANO_TO_MHR_HAND20_ORDER = [
    4, 3, 2, 1,    # thumb: tip -> base
    8, 7, 6, 5,    # index: tip -> base
    12, 11, 10, 9, # middle: tip -> base
    16, 15, 14, 13,# ring: tip -> base
    20, 19, 18, 17 # pinky: tip -> base
]
```

## 5) Practical Integration Notes

- If your external hand detector predicts both hands separately, map each side with the corresponding list above.
- If your image is mirrored/flipped in preprocessing, handle left/right swap before mapping.
- If you only replace finger points and keep body detector wrist, you can skip MANO index `0`.

## 6) Sources Checked

- Local repo: `mhr70.py`
- HaMeR MANO wrapper: `https://github.com/geopavlakos/hamer/blob/main/hamer/models/mano_wrapper.py`
- WiLoR MANO wrapper: `https://github.com/rolpotamias/WiLoR/blob/main/wilor/models/mano_wrapper.py`
