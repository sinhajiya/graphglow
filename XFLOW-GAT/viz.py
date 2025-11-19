import numpy as np
import cv2

def viz(
        img_non, img_make,
        coordsN, coordsM,
        roi_labelsN, roi_labelsM,
        edge_index, is_make,
        out_path
    ):

    coordsN = np.asarray(coordsN, dtype=np.float32)
    coordsM = np.asarray(coordsM, dtype=np.float32)
    roi_labels = np.concatenate([roi_labelsN, roi_labelsM])
    is_make = np.asarray(is_make)

    H, W = img_non.shape[:2]
    canvas = np.zeros((H, W*2, 3), dtype=np.uint8)
    canvas[:, :W]  = img_non
    canvas[:, W:]  = img_make

    coordsM_shift = coordsM.copy()
    coordsM_shift[:,0] += W
    coords_all = np.vstack([coordsN, coordsM_shift])

    roi_mask = roi_labels != 0
    e = edge_index
    keep = roi_mask[e[0]] & roi_mask[e[1]]
    e = e[:, keep]

    for i in range(e.shape[1]):
        u = int(e[0, i]); v = int(e[1, i])
        x1, y1 = coords_all[u]; x2, y2 = coords_all[v]
        pt1 = (int(x1), int(y1)); pt2 = (int(x2), int(y2))
        if is_make[u] == 0 and is_make[v] == 0:
            color = (255, 255, 255)
        elif is_make[u] == 1 and is_make[v] == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.line(canvas, pt1, pt2, color, 1)

    for i, (x, y) in enumerate(coords_all):
        if not roi_mask[i]: continue
        pt = (int(x), int(y))
        if is_make[i] == 0:
            cv2.circle(canvas, pt, 2, (255, 255, 0), -1)
        else:
            cv2.circle(canvas, pt, 2, (0, 255, 255), -1)

    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
