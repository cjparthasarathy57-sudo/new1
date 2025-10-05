# app.py
from flask import Flask, request, send_file, jsonify, safe_join
import os, tempfile, zipfile
from processing import process_image_bytes

app = Flask(__name__)

@app.route("/api/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/api/process", methods=["POST"])
def process_endpoint():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    use_dl = request.form.get('use_dl', '0') == '1'
    k = int(request.form.get('k', 3))
    downscale = int(request.form.get('downscale', 1024))

    # create temporary dir to store outputs
    tmp = tempfile.mkdtemp(prefix="floorplan_")
    try:
        res = process_image_bytes(file.stream, use_dl=use_dl, k=k, downscale=downscale, out_dir=tmp)
        # write preview png from buffer
        preview_path = os.path.join(tmp, "preview.png")
        with open(preview_path, "wb") as f:
            f.write(res["preview_png_buf"].getbuffer())

        # zip outputs
        zip_path = os.path.join(tmp, "results.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(preview_path, arcname="preview.png")
            z.write(res["svg_path"], arcname="plan.svg")
            z.write(res["dxf_path"], arcname="plan.dxf")

        # respond with zip file stream
        return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name="floorplan_results.zip")
    except Exception as e:
        import traceback; traceback.print_exc()
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
