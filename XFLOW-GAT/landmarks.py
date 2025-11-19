
import mediapipe as mp
import numpy as np
import cv2


from face_land import FACE_REGIONS, ROI_LABELS

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1
)

def extract_landmark_info(image_rgb):
    H, W = image_rgb.shape[:2]
    res = mp_face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None, None
    lm = res.multi_face_landmarks[0].landmark
    coords = np.array([[int(l.x * W), int(l.y * H)] for l in lm], dtype=np.int32)
    labels = np.zeros(len(coords), dtype=np.int64)
    for r, ids in FACE_REGIONS.items():
        lab = ROI_LABELS[r]
        for i in ids:
            if i < len(labels): labels[i] = lab
    return coords, labels

def face_parsing_mask_fallback(coords, H, W, blur_ks=31, blur_sigma=9.0):
    masks, face_mask = {}, np.zeros((H, W), np.float32)
    for region, ids in FACE_REGIONS.items():
        pts = [coords[i] for i in ids if i < len(coords)]
        if len(pts) < 3:
            masks[region] = np.zeros((H, W), np.float32)
            continue
        hull = cv2.convexHull(np.array(pts))
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        m = cv2.GaussianBlur(m.astype(np.float32), (blur_ks, blur_ks), blur_sigma)
        masks[region] = np.clip(m, 0, 1)
        face_mask = np.clip(face_mask + m, 0, 1)
    masks["face"] = face_mask
    return masks
