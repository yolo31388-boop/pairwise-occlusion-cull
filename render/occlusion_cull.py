"""GPU-style Hi-Z occlusion culling.

Fixed defects vs. the broken version:
- build_hiz builds the *complete* max-pooled pyramid down to 1x1
  (was: single level, and compared against the previous frame's mip).
- is_occluded tests the AABB against the farthest occluder depth over
  its whole screen footprint (conservative nearest-depth test).
- camera_changed detects camera motion so stale previous-frame
  occlusion results are invalidated instead of reused.
- cull_objects keeps transparent/particle objects in a separate pass:
  they don't write depth and must never be depth-culled.
- OcclusionCuller builds one Hi-Z pyramid per view (VR/multi-view safe)
  and temporally reprojects results that arrive with readback latency.
"""

import math

__all__ = [
    'build_hiz',
    'project_aabb',
    'is_occluded',
    'camera_changed',
    'cull_objects',
    'OcclusionCuller',
]


def build_hiz(depth, width, height):
    """Build the full Hi-Z max pyramid from a depth buffer.

    Returns a list of mip levels. Level 0 is the source depth buffer;
    each subsequent level is the 2x2 max of the previous one, down to
    1x1. Storing the *max* (farthest) depth per texel is what makes the
    occlusion test conservative: an object may only be culled when its
    nearest point is behind the farthest occluder in the covered area.
    """
    levels = [[list(row) for row in depth]]
    w, h = width, height
    while w > 1 or h > 1:
        prev = levels[-1]
        nw, nh = max(1, (w + 1) // 2), max(1, (h + 1) // 2)
        level = []
        for y in range(nh):
            row = []
            for x in range(nw):
                m = prev[2 * y][2 * x]
                if 2 * x + 1 < w:
                    m = max(m, prev[2 * y][2 * x + 1])
                if 2 * y + 1 < h:
                    m = max(m, prev[2 * y + 1][2 * x])
                    if 2 * x + 1 < w:
                        m = max(m, prev[2 * y + 1][2 * x + 1])
                row.append(m)
            level.append(row)
        levels.append(level)
        w, h = nw, nh
    return levels


def _transform_point(m, x, y, z):
    """Multiply a row-major 4x4 matrix by a column vec4."""
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
        m[12] * x + m[13] * y + m[14] * z + m[15],
    )


def _project_aabb_ndc(aabb, view_proj):
    """Project an AABB to NDC as (x0, y0, zmin, x1, y1, zmax).

    Returns None if any corner crosses the camera plane (w <= 0); such
    a box cannot be safely clipped and must be treated as visible.
    """
    minx, miny, minz, maxx, maxy, maxz = aabb
    x0 = y0 = zmin = float('inf')
    x1 = y1 = zmax = float('-inf')
    for cx in (minx, maxx):
        for cy in (miny, maxy):
            for cz in (minz, maxz):
                px, py, pz, pw = _transform_point(view_proj, cx, cy, cz)
                if pw <= 1e-6:
                    return None
                nx, ny, nz = px / pw, py / pw, pz / pw
                x0, x1 = min(x0, nx), max(x1, nx)
                y0, y1 = min(y0, ny), max(y1, ny)
                zmin, zmax = min(zmin, nz), max(zmax, nz)
    return (x0, y0, zmin, x1, y1, zmax)


def project_aabb(aabb, view_proj, screen):
    """Project an AABB to a clipped screen rect (x, y, width, height).

    NDC [-1, 1] maps to [0, screen] on both axes (no y-flip); the rect
    is clamped to the viewport. A box crossing the camera plane covers
    the whole screen (conservative: never culled).
    """
    w, h = screen
    ndc = _project_aabb_ndc(aabb, view_proj)
    if ndc is None:
        return (0.0, 0.0, float(w), float(h))
    x0, y0, _, x1, y1, _ = ndc
    px0 = max(0.0, min((x0 + 1.0) * 0.5 * w, float(w)))
    py0 = max(0.0, min((y0 + 1.0) * 0.5 * h, float(h)))
    px1 = max(0.0, min((x1 + 1.0) * 0.5 * w, float(w)))
    py1 = max(0.0, min((y1 + 1.0) * 0.5 * h, float(h)))
    return (px0, py0, px1 - px0, py1 - py0)


def is_occluded(rect, hiz, width, height, conservative=False):
    """Test a screen-space box against the Hi-Z pyramid.

    rect is (x0, y0, zmin, x1, y1, zmax): x/y in pixels, z in the same
    depth space stored in the pyramid. The box is occluded only if its
    test depth is behind the farthest occluder depth over its whole
    footprint. By default the test depth is zmin (the box's nearest
    point); with conservative=True zmax is used, so a box is only
    reported occluded when even its farthest point is hidden.
    """
    x0, y0, zmin, x1, y1, zmax = rect
    x0 = max(0.0, min(x0, float(width)))
    x1 = max(0.0, min(x1, float(width)))
    y0 = max(0.0, min(y0, float(height)))
    y1 = max(0.0, min(y1, float(height)))
    if x1 <= x0 or y1 <= y0:
        return False  # zero-area footprint: nothing to cull

    # Pick the first mip whose texels roughly cover the box footprint.
    box_w, box_h = x1 - x0, y1 - y0
    level = 0
    while level < len(hiz) - 1:
        lw, lh = len(hiz[level][0]), len(hiz[level])
        if box_w <= 2.0 * width / lw and box_h <= 2.0 * height / lh:
            break
        level += 1
    mip = hiz[level]
    lw, lh = len(mip[0]), len(mip)

    tx0 = min(lw - 1, int(x0 / width * lw))
    ty0 = min(lh - 1, int(y0 / height * lh))
    tx1 = min(lw - 1, max(tx0, int(math.ceil(x1 / width * lw)) - 1))
    ty1 = min(lh - 1, max(ty0, int(math.ceil(y1 / height * lh)) - 1))

    occluder = float('-inf')
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            occluder = max(occluder, mip[ty][tx])

    test_depth = zmax if conservative else zmin
    return test_depth > occluder


def camera_changed(prev, curr, threshold):
    """True if any view-projection element moved by >= threshold.

    Used to invalidate previous-frame occlusion results: reusing them
    after camera motion causes edge flicker (objects vanish for a frame
    and pop back).
    """
    return any(abs(a - b) >= threshold for a, b in zip(prev, curr))


def cull_objects(objects, hiz, view_proj, screen):
    """Return the ids of visible objects, preserving input order.

    Transparent and particle objects are kept unconditionally: they do
    not write depth, so depth-culling them is a bug — they belong to a
    separate pass. Boxes crossing the camera plane are kept as well.
    """
    w, h = screen
    visible = []
    for obj in objects:
        if obj.get('transparent') or obj.get('particle'):
            visible.append(obj['id'])
            continue
        ndc = _project_aabb_ndc(obj['aabb'], view_proj)
        if ndc is None:
            visible.append(obj['id'])
            continue
        x0, y0, zmin, x1, y1, zmax = ndc
        rect = (
            (x0 + 1.0) * 0.5 * w,
            (y0 + 1.0) * 0.5 * h,
            zmin,
            (x1 + 1.0) * 0.5 * w,
            (y1 + 1.0) * 0.5 * h,
            zmax,
        )
        if not is_occluded(rect, hiz, w, h):
            visible.append(obj['id'])
    return visible


class OcclusionCuller:
    """Stateful multi-view occlusion culling with temporal reprojection.

    - One Hi-Z pyramid per view: VR eyes / split-screen views never
      share a pyramid, so the right eye stops losing objects.
    - Previous-frame results are invalidated whenever the camera moves
      (camera_changed), so fast pans don't reuse stale culling.
    - GPU readback results arrive `readback_latency` frames late; each
      pending result is reprojected against the camera that generated
      it and only applied if that camera still matches the current one,
      otherwise the object is treated as visible (no disappear/pop).
    """

    def __init__(self, num_views=1, readback_latency=2, camera_threshold=1e-3):
        self.num_views = num_views
        self.readback_latency = readback_latency
        self.camera_threshold = camera_threshold
        self.hiz = [None] * num_views
        self.cameras = [None] * num_views
        self._pending = []  # (frames_left, view, camera, results)

    def update_view(self, view, depth, width, height, camera):
        """Rebuild one view's pyramid from *this* frame's depth buffer."""
        if self.cameras[view] is None or camera_changed(
            self.cameras[view], camera, self.camera_threshold
        ):
            # Camera moved: drop stale in-flight results for this view.
            self._pending = [p for p in self._pending if p[1] != view]
        self.hiz[view] = build_hiz(depth, width, height)
        self.cameras[view] = camera

    def submit(self, view, results):
        """Queue a culling result set that lands after the readback delay."""
        self._pending.append((self.readback_latency, view, self.cameras[view], results))

    def collect(self):
        """Advance one frame and return reprojected results that are still valid."""
        ready = []
        still_pending = []
        for frames_left, view, camera, results in self._pending:
            frames_left -= 1
            if frames_left > 0:
                still_pending.append((frames_left, view, camera, results))
                continue
            # Temporal reprojection check: only apply if the camera that
            # produced the result still matches the view's camera.
            if self.cameras[view] is not None and not camera_changed(
                camera, self.cameras[view], self.camera_threshold
            ):
                ready.append((view, results))
        self._pending = still_pending
        return ready
