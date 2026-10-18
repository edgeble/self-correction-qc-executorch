# 70 calibration frames from the two hackathon videos: letterboxed 640x640, RGB 0..1, float32 NCHW raw
import cv2, numpy as np, os, sys
out = sys.argv[1]; vids = sys.argv[2:]; os.makedirs(out, exist_ok=True); k = 0
for v in vids:
    cap = cv2.VideoCapture(v); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = max(1, n // 35)
    for i in range(0, n, step):
        if k >= 70: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
        if not ok: continue
        h, w = fr.shape[:2]; r = 640 / max(h, w); nw, nh = int(w * r), int(h * r)
        img = np.full((640, 640, 3), 114, np.uint8); rs = cv2.resize(fr, (nw, nh))
        y0, x0 = (640 - nh) // 2, (640 - nw) // 2; img[y0:y0+nh, x0:x0+nw] = rs
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x.transpose(2, 0, 1)[None].tofile(os.path.join(out, 'a_%03d.raw' % (k + 1))); k += 1
    cap.release()
print('frames', k)
