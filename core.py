import os, cv2, math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Box:
    frame: int
    t_ms: int
    id: int
    x: int
    y: int
    w: int
    h: int
    source: str = "auto"  # "auto" or "manual"

# ---------- utils ----------
def ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_box(x, y, w, h, pad_x: float, pad_y: float, W: int, H: int):
    x_new = int(round(x - w * pad_x))
    y_new = int(round(y - h * pad_y))
    w_new = int(round(w * (1 + 2 * pad_x)))
    h_new = int(round(h * (1 + 2 * pad_y)))
    x_new = clamp(x_new, 0, W - 1)
    y_new = clamp(y_new, 0, H - 1)
    w_new = clamp(w_new, 1, W - x_new)
    h_new = clamp(h_new, 1, H - y_new)
    return x_new, y_new, w_new, h_new

# ---------- mosaic ops ----------
def pixelate_region(img, x, y, w, h, strength: int):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    s = max(1, strength)
    small_w = max(1, w // s)
    small_h = max(1, h // s)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = mosaic

def gaussian_blur_region(img, x, y, w, h, strength: int):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    k = max(1, strength)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y:y+h, x:x+w] = blurred

# ---------- simple tracker ----------
def assign_ids(prev_boxes: List[Box], curr_rects: List[Tuple[int,int,int,int]], frame_idx: int, t_ms: int) -> List[Box]:
    assigned: List[Box] = []
    used_prev = set()
    next_id_base = (prev_boxes[-1].id + 1) if prev_boxes else 0

    def center(x,y,w,h): return (x + w/2.0, y + h/2.0)

    for (x,y,w,h) in curr_rects:
        cx, cy = center(x,y,w,h)
        best = None
        best_d = 1e9
        for i, pb in enumerate(prev_boxes):
            if i in used_prev:
                continue
            pcx, pcy = center(pb.x, pb.y, pb.w, pb.h)
            d = math.hypot(cx - pcx, cy - pcy)
            if d < best_d:
                best_d = d
                best = (i, pb)
        th = max(w, h) * 1.5
        if best is not None and best_d < th:
            i, pb = best
            used_prev.add(i)
            assigned.append(Box(frame_idx, t_ms, pb.id, x,y,w,h))
        else:
            assigned.append(Box(frame_idx, t_ms, next_id_base, x,y,w,h))
            next_id_base += 1
    return assigned

