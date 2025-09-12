import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

# ------------------ Toroidal helpers ------------------

class _DSU:
    def __init__(self, n):
        self.parent = np.arange(n + 1, dtype=np.int32)
        self.rank = np.zeros(n + 1, dtype=np.int8)
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def _remap_sequential(labels):
    if labels.size == 0: return labels
    uniq = np.unique(labels); uniq = uniq[uniq != 0]
    if uniq.size == 0: return labels
    mapping = {old: i + 1 for i, old in enumerate(uniq)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = new
    return out

def toroidal_delta(old, new, size):
    d = new - old
    if d > size / 2: d -= size
    elif d < -size / 2: d += size
    return d

def toroidal_distance(a, b, H, W):
    dr = toroidal_delta(a[0], b[0], H)
    dc = toroidal_delta(a[1], b[1], W)
    return np.hypot(dr, dc)

def fast_toroidal_weighted_centroid(rows, cols, weights, H, W):
    """
    An optimized, arithmetic-only version of toroidal centroid calculation.
    It avoids trigonometry by 'unwrapping' coordinates relative to a stable anchor point.
    """
    if rows.size == 0:
        return np.array([0.0, 0.0])

    # FIX: Use the first point as a stable anchor. The arithmetic mean is unreliable
    # when points are already wrapped across a boundary.
    r_anchor = rows[0]
    c_anchor = cols[0]

    # Unwrap coordinates relative to the anchor using modular arithmetic properties.
    # This is a concise and robust way to handle the wrapping.
    unwrapped_rows = rows - H * np.round((rows - r_anchor) / H)
    unwrapped_cols = cols - W * np.round((cols - c_anchor) / W)
    
    # Perform a standard weighted average on the unwrapped coordinates.
    r_c = np.average(unwrapped_rows, weights=weights)
    c_c = np.average(unwrapped_cols, weights=weights)

    # Wrap the final result back into the toroidal space.
    return np.array([r_c % H, c_c % W], dtype=np.float64)

def _union_wrap_seams(labels, min_size=1):
    H, W = labels.shape
    n = int(labels.max())
    if n == 0:
        return labels.astype(np.int32)

    dsu = _DSU(n)

    for r in range(H):
        a = int(labels[r, 0])
        if a > 0:
            for dr in (-1, 0, 1):
                rr = (r + dr) % H
                b = int(labels[rr, W - 1])
                if b > 0:
                    dsu.union(a, b)

    for c in range(W):
        a = int(labels[0, c])
        if a > 0:
            for dc in (-1, 0, 1):
                cc = (c + dc) % W
                b = int(labels[H - 1, cc])
                if b > 0:
                    dsu.union(a, b)

    map_root = np.arange(n + 1, dtype=np.int32)
    for lab in range(1, n + 1):
        map_root[lab] = dsu.find(lab)
    map_root[0] = 0
    roots_img = map_root[labels]

    if int(min_size) > 1:
        counts = np.bincount(roots_img.ravel(), minlength=map_root.max() + 1)
        keep_root = counts >= int(min_size)
        new_id = np.zeros_like(keep_root, dtype=np.int32)
        cid = 1
        for rid, keep in enumerate(keep_root):
            if rid == 0: continue
            if keep:
                new_id[rid] = cid
                cid += 1
        out = new_id[roots_img]
    else:
        out = _remap_sequential(roots_img)

    return out.astype(np.int32)

def toroidal_segment(binary_map, mode="label", peak_distance=7, min_size=1):
    H, W = binary_map.shape
    if not np.any(binary_map):
        return np.zeros((H, W), dtype=np.int32)

    if mode == "watershed":
        big_mask = np.tile(binary_map.astype(bool), (3, 3))
        distance = distance_transform_edt(big_mask)
        coords = peak_local_max(distance, min_distance=int(peak_distance), labels=big_mask)
        peak_mask = np.zeros_like(distance, dtype=bool)
        if coords.size > 0:
            peak_mask[tuple(coords.T)] = True
        markers, _ = measure.label(peak_mask, return_num=True, connectivity=2)
        labels_big = watershed(-distance, markers, mask=big_mask)
        labels_center = labels_big[H:2*H, W:2*W].astype(np.int32)
        labels_final = _union_wrap_seams(labels_center, min_size=min_size)
        return labels_final

    labels_center, num_labels = measure.label(binary_map.astype(bool), connectivity=2, return_num=True)
    if num_labels == 0:
        return labels_center.astype(np.int32)
    labels_final = _union_wrap_seams(labels_center.astype(np.int32), min_size=min_size)
    return labels_final