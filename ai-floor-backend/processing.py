# processing.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString
import svgwrite
import ezdxf
import io
import os
import base64

# Optional imports for deep learning segmentation
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models.segmentation import fcn_resnet50
    DL_AVAILABLE = True
except Exception:
    DL_AVAILABLE = False

def load_image_bytes(stream):
    # stream is a file-like object
    arr = np.frombuffer(stream.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def preprocess(img, max_side=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def hsv_threshold(img):
    # convert to HSV and attempt to find darker (lines) regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # thresholds tuned for dark line detection — adjust as needed
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 150])
    mask = cv2.inRange(hsv, lower, upper)
    # morphological cleanups
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def kmeans_segment(img, k=3):
    # color clustering (for colored scanned plans)
    data = img.reshape((-1,3)).astype(np.float32)
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    labels = km.labels_.reshape(img.shape[:2])
    return labels

def find_contours_from_mask(mask):
    # find binary contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 100:  # ignore tiny
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) >= 3:
            polys.append(pts)
    return polys

def dl_segmentation_mask(img):
    # use torchvision FCN pretrained model (coarse) to extract masks — optional
    if not DL_AVAILABLE:
        raise RuntimeError("PyTorch/TorchVision not installed")
    model = fcn_resnet50(pretrained=True).eval()
    # transform
    transform = T.Compose([T.ToTensor()])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    # pick highest class probabilities (we are not fine-tuned for floorplans, but it can help)
    mask = torch.argmax(output, dim=0).byte().cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    # resize to original img size if necessary
    return mask

def vectorize_polygons(polygons):
    # convert list of point lists into shapely Polygons and polylines normalized
    shapes = []
    for pts in polygons:
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area > 10:
                shapes.append(poly)
        except Exception:
            continue
    return shapes

def make_preview_png(img, polygons):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, 'RGBA')
    for poly in polygons:
        xy = list(poly.exterior.coords)
        draw.line(xy + [xy[0]], fill=(255,0,0,200), width=3)
        draw.polygon(xy, outline=(255,0,0,180), fill=None)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

def save_svg(polygons, size, out_path):
    w, h = size
    dwg = svgwrite.Drawing(out_path, size=(w,h))
    for poly in polygons:
        pts = [(x, y) for x,y in poly.exterior.coords]
        dwg.add(dwg.polyline(pts, stroke="black", fill="none", stroke_width=2))
    dwg.save()

def save_dxf(polygons, out_path):
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for poly in polygons:
        pts = [(float(x), float(y)) for x,y in poly.exterior.coords]
        # create LWPOLYLINE
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, close=True)
    doc.saveas(out_path)

def process_image_bytes(stream, use_dl=False, k=3, downscale=1024, out_dir="/tmp"):
    img = load_image_bytes(stream)
    orig_shape = (img.shape[1], img.shape[0])
    img = preprocess(img, max_side=downscale)

    # Step A: HSV threshold
    mask_hsv = hsv_threshold(img)

    # Step B: K-means segmentation (if colored)
    labels = kmeans_segment(img, k=k)
    # pick a cluster that is darkest on average
    cluster_means = []
    for i in range(k):
        mean_val = img[labels==i].mean() if (labels==i).any() else 255
        cluster_means.append(mean_val)
    darkest_cluster = int(np.argmin(cluster_means))
    mask_k = (labels==darkest_cluster).astype(np.uint8)*255

    # Combine masks
    combined = cv2.bitwise_or(mask_hsv, mask_k)
    # remove small noisy bits
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    combined = cv2.dilate(combined, np.ones((3,3), np.uint8), iterations=1)

    # Optional deep learning mask for difficult images
    if use_dl and DL_AVAILABLE:
        try:
            mask_dl = dl_segmentation_mask(img)
            combined = cv2.bitwise_or(combined, mask_dl)
        except Exception as e:
            print("DL segmentation failed:", e)

    # find contours
    polys = find_contours_from_mask(combined)
    shapes = vectorize_polygons(polys)

    # generate outputs
    preview_buf = make_preview_png(img, shapes)

    svg_path = os.path.join(out_dir, "plan.svg")
    save_svg(shapes, (img.shape[1], img.shape[0]), svg_path)

    dxf_path = os.path.join(out_dir, "plan.dxf")
    save_dxf(shapes, dxf_path)

    # Package into memory dict with base64 or paths
    result = {
        "preview_png_buf": preview_buf,
        "svg_path": svg_path,
        "dxf_path": dxf_path,
        "polygons_count": len(shapes),
        "size": (img.shape[1], img.shape[0])
    }
    return result
