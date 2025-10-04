import base64
import io
import csv
import random
import json as _json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import numpy as np
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from ultralytics import YOLO

# ---- YENİ: urllib ile NASA Images API proxy (requests eklemeden) ----
import urllib.request
import urllib.parse

APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_ROOT / "models"
TEMP_DIR = APP_ROOT / "app" / "static" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "best.pt"
CLASS_NAMES = ["No Exoplanet", "Exoplanet Candidate"]

app = Flask(__name__)

# ---------------------------
# (Var olan) yardımcılar — KISALTILMIŞ
# ---------------------------
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model weights not found at models/best.pt. Train first.")
    return YOLO(str(MODEL_PATH))

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_pil(b64_uri: str) -> Image.Image:
    b64 = b64_uri.split(",", 1)[1] if "," in b64_uri else b64_uri
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def normalize_name(s: str) -> str:
    return (s or "").strip()

def archive_url_for(hostname: str) -> Optional[str]:
    if not hostname: return None
    return f"https://exoplanetarchive.ipac.caltech.edu/cgi-bin/DisplayOverview/nph-DisplayOverview?objname={hostname.replace(' ', '+')}&type=CONFIRMED_PLANETS"

def is_nasa_confirmed_transiting(hostname: str):
    info = {"hostname": hostname, "discoverymethod": None, "disc_year": None, "facility": None, "planet_count": None}
    try:
        table = NasaExoplanetArchive.query_criteria(table="pscomppars", select="*", where=f"hostname='{hostname}'")
    except Exception:
        return False, info
    if len(table) == 0:
        return False, info
    
    import numpy as _np
    trans_col = None
    for col_name in ["tran_flag", "discoverymethod"]:
        if col_name in table.colnames:
            if col_name == "tran_flag":
                trans_col = table[col_name]
                break
            elif col_name == "discoverymethod":
                methods = [str(m).lower() for m in table[col_name]]
                if any("transit" in m for m in methods):
                    confirmed = True
                else:
                    confirmed = False
                break
    
    if trans_col is not None:
        confirmed = bool((_np.array(trans_col) == 1).any())
    else:
        confirmed = False
    
    row0 = table[0]
    info["planet_count"] = int(len(table))
    info["discoverymethod"] = str(row0.get("discoverymethod") or "") if "discoverymethod" in table.colnames else ""
    info["disc_year"] = int(row0.get("disc_year") or 0) if ("disc_year" in table.colnames and row0.get("disc_year") is not None) else None
    info["facility"] = str(row0.get("disc_facility") or "") if "disc_facility" in table.colnames else ""
    return confirmed, info

def fetch_star_quick_facts(target: str, mission: str) -> Dict[str, Any]:
    target = normalize_name(target)
    facts = {
        "target": target, "mission": mission, "exists_in_mast": False,
        "ra": None, "dec": None, "mag": None, "nasa_confirmed": False,
        "discovery": {"method": None, "year": None, "facility": None, "who": None},
        "archive_url": archive_url_for(target)
    }
    try:
        sr = lk.search_lightcurve(target, mission=mission)
    except Exception:
        sr = []
    if len(sr) > 0:
        facts["exists_in_mast"] = True
        try:
            meta = sr.table[0] if hasattr(sr, "table") and len(sr.table) > 0 else None
            if meta is not None:
                for cand in ("ra","RA","target_ra","s_ra"):
                    if cand in meta.colnames: 
                        try:
                            facts["ra"] = float(meta[cand])
                            break
                        except:
                            pass
                for cand in ("dec","DEC","target_dec","s_dec"):
                    if cand in meta.colnames: 
                        try:
                            facts["dec"] = float(meta[cand])
                            break
                        except:
                            pass
                if mission == "Kepler":
                    for cand in ("kic_kepmag","KIC_kepmag","kepmag"):
                        if cand in meta.colnames:
                            try: 
                                facts["mag"] = float(meta[cand])
                                break
                            except: 
                                pass
                elif mission == "TESS":
                    for cand in ("tess_mag","Tmag","tmag"):
                        if cand in meta.colnames:
                            try: 
                                facts["mag"] = float(meta[cand])
                                break
                            except: 
                                pass
        except Exception:
            pass
    confirmed, conf = is_nasa_confirmed_transiting(target)
    facts["nasa_confirmed"] = confirmed
    facts["discovery"]["method"] = conf.get("discoverymethod")
    facts["discovery"]["year"] = conf.get("disc_year")
    facts["discovery"]["facility"] = conf.get("facility")
    facts["discovery"]["who"] = "NASA Confirmed" if confirmed else "User Candidate"
    return facts

def quick_lightcurve_pair(target: str, mission: str):
    try:
        sr = lk.search_lightcurve(target, mission=mission)
        if len(sr) == 0: return None, None
        lc = sr[0].download()
        if lc is None: return None, None
        try: 
            lc = lc.remove_nans().flatten(window_length=401)
        except Exception: 
            pass
        
        # raw
        import matplotlib.pyplot as plt
        ax = lc.normalize().plot()
        fig = ax.figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        raw_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # phase (naive BLS)
        phase_b64 = None
        try:
            from astropy.timeseries import BoxLeastSquares
            time = lc.time.value
            flux = lc.normalize().flux.value
            import numpy as _np
            mask = _np.isfinite(time) & _np.isfinite(flux)
            time, flux = time[mask], flux[mask]
            if len(time) > 100:
                durations = _np.linspace(0.05, 0.2, 10) * (time.max()-time.min())/50.0
                periods = _np.linspace(0.5, 20.0, 500)
                bls = BoxLeastSquares(time, flux)
                power = bls.power(periods, durations)
                i = int(power.power.argmax())
                period, t0 = power.period[i], power.transit_time[i]
                lc_fold = lc.fold(period=period, epoch_time=t0)
                ax2 = lc_fold.bin(bins=200).plot(marker="o", linestyle="none", markersize=3, color="white")
                fig2 = ax2.figure
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig2)
                phase_b64 = "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Phase fold error: {e}")
            phase_b64 = None
        
        return raw_b64, phase_b64
    except Exception as e:
        print(f"Light curve error: {e}")
        return None, None

def run_model_on_pil(img: Image.Image) -> Dict[str, Any]:
    model = load_model()
    res = model.predict(img, verbose=False)
    r0 = res[0]
    top_idx = int(r0.probs.top1)
    conf = float(r0.probs.top1conf) * 100.0
    probs = r0.probs.data.tolist() if hasattr(r0.probs, "data") else []
    prob_pos = float(probs[1] * 100.0) if len(probs) > 1 else (conf if top_idx == 1 else 100.0 - conf)
    prob_neg = float(probs[0] * 100.0) if len(probs) > 0 else (conf if top_idx == 0 else 100.0 - conf)
    return {
        "prediction_label": CLASS_NAMES[top_idx],
        "confidence": round(conf, 2),
        "prob_positive": round(prob_pos, 2),
        "prob_negative": round(prob_neg, 2)
    }

# ---------------------------
# ROUTES (mevcutlara ek)
# ---------------------------
@app.route("/")
def root():
    """Ana sayfa"""
    return render_template("index.html")

@app.get("/starinfo")
def starinfo():
    target = normalize_name(request.args.get("target", ""))
    mission = (request.args.get("mission") or "auto").strip()
    if not target: 
        return jsonify({"error": "target is required"}), 400
    if mission.lower() == "auto":
        try:
            if len(lk.search_lightcurve(target, mission="Kepler")) > 0: 
                mission = "Kepler"
            elif len(lk.search_lightcurve(target, mission="TESS")) > 0: 
                mission = "TESS"
            else: 
                mission = "Kepler"
        except Exception:
            mission = "Kepler"
    facts = fetch_star_quick_facts(target, mission)
    return jsonify(facts)

@app.post("/analyze")
def analyze():
    try:
        if "file" in request.files:
            img = Image.open(request.files["file"].stream).convert("RGB")
            pred = run_model_on_pil(img)
            return jsonify({
                **pred, "source": "upload", "plot": pil_to_b64(img),
                "star_info": None
            })
        data = request.get_json(silent=True) or {}
        if "image_b64" in data and data["image_b64"]:
            img = b64_to_pil(data["image_b64"])
            pred = run_model_on_pil(img)
            return jsonify({**pred, "source": "upload_b64", "plot": pil_to_b64(img), "star_info": None})
        
        target_name = normalize_name(data.get("target_name", ""))
        mission = (data.get("mission") or "auto").strip()
        if not target_name: 
            return jsonify({"error":"Provide an image or a target_name."}), 400
        
        if mission.lower() == "auto":
            try:
                if len(lk.search_lightcurve(target_name, mission="Kepler")) > 0: 
                    mission = "Kepler"
                elif len(lk.search_lightcurve(target_name, mission="TESS")) > 0: 
                    mission = "TESS"
                else: 
                    mission = "Kepler"
            except Exception:
                mission = "Kepler"
        
        raw_b64, phase_b64 = quick_lightcurve_pair(target_name, mission)
        pred = {"prediction_label":"N/A","confidence":None,"prob_positive":None,"prob_negative":None}
        if phase_b64:
            pred = run_model_on_pil(b64_to_pil(phase_b64))
        elif raw_b64:
            pred = run_model_on_pil(b64_to_pil(raw_b64))
            
        star_info = fetch_star_quick_facts(target_name, mission)
        return jsonify({
            **pred, "source":"target",
            "plots":{"raw":raw_b64,"phase":phase_b64},
            "star_info": star_info, "mission_badge": mission
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {e}"}), 500

# ---- NASA Images API proxy (asset yok, canlı çekim) ----
@app.get("/nasa/images")
def nasa_images():
    """
    Proxy to NASA Images API (https://images-api.nasa.gov/search?q=...).
    Params: ?q=exoplanet&media_type=image&n=12
    Returns simplified list: [{"title":...,"href":...,"thumb":...}]
    """
    q = request.args.get("q", "exoplanet")
    n = int(request.args.get("n", "12"))
    params = {"q": q, "media_type": "image"}
    url = "https://images-api.nasa.gov/search?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"NASA API fetch failed: {e}"}), 502

    items = (data.get("collection", {}) or {}).get("items", [])[:max(1, n)]
    out = []
    for it in items:
        d = (it.get("data") or [{}])[0]
        l = (it.get("links") or [{}])[0]
        title = d.get("title", "NASA Image")
        thumb = l.get("href")
        href = thumb
        out.append({"title": title, "href": href, "thumb": thumb})
    return jsonify(out)

@app.get("/static/temp/<path:filename>")
def tempfiles(filename):
    return send_from_directory(TEMP_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    print("=" * 60)
    print("  NASA EXOPLANET DETECTOR - Live Cosmos Edition")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Model exists: {MODEL_PATH.exists()}")
    print(f"  Server: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
