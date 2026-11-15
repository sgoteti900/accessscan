# ============ PART A: imports and settings ============
from datetime import datetime
from html import escape
from pathlib import Path

import folium
import gspread
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_folium import st_folium
from torch import nn
from torchvision import models, transforms

st.set_page_config(page_title="AccessScan", page_icon="♿", layout="wide")

HERE = Path(__file__).parent      # the folder this file is in
LABELS = ["ramp", "stairs_only", "level_entrance", "door_width", "auto_opener"]
NICE = {"ramp": "Ramp", "stairs_only": "Stairs only", "level_entrance": "Level entrance",
        "door_width": "Wide door", "auto_opener": "Auto-opener"}


# ============ PART B: the model (must match training exactly) ============
class PadToSquare:
    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        return ImageOps.expand(image, border=((side-width)//2, (side-height)//2,
                               side-width-(side-width)//2, side-height-(side-height)//2),
                               fill=(124, 116, 104))


eval_transform = transforms.Compose([
    PadToSquare(), transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_model():
    network = models.mobilenet_v3_small(weights=None)
    features = network.classifier[0].in_features
    network.classifier = nn.Sequential(nn.Linear(features, 128), nn.Hardswish(),
                                       nn.Dropout(p=0.3), nn.Linear(128, len(LABELS)))
    return network


@st.cache_resource
def load_model():
    saved = torch.load(HERE / "accessscan_mobilenet_v3_small.pt",
                       map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["threshold"]


model, THRESHOLD = load_model()


def predict(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    x = eval_transform(image).unsqueeze(0)
    with torch.inference_mode():
        scores = torch.sigmoid(model(x))[0].tolist()
    features = {label: int(s >= THRESHOLD) for label, s in zip(LABELS, scores)}
    return features, dict(zip(LABELS, scores))


# ============ PART C: the scoring rule (same as the notebook) ============
def accessibility_score(features):
    ramp, stairs, level = (bool(features[k]) for k in ["ramp", "stairs_only", "level_entrance"])
    if stairs and (ramp or level):
        return None, ["Conflicting route predictions: stairs-only and a ramp/level route. Review the photo."]
    if not (ramp or stairs or level):
        return None, ["No clear route predicted. A closer or wider photo and human review are needed."]
    if level:
        points, reasons = 60, ["Visible level route predicted: +60 route points."]
    elif ramp:
        points, reasons = 50, ["Ramp predicted: +50 route points; slope and condition are not assessed."]
    else:
        points, reasons = 0, ["Stairs-only predicted: +0 route points. This is a stair barrier."]
    if features["door_width"]:
        points += 25
        reasons.append("Wide door predicted: +25; visual estimate, not a clearance measurement.")
    else:
        reasons.append("Narrow door predicted: +0; verify with a measurement.")
    if features["auto_opener"]:
        points += 15
        reasons.append("Auto-opener predicted: +15; operation is not verified.")
    else:
        reasons.append("No auto-opener predicted: +0; a button could be out of frame.")
    if stairs and points > 20:
        points = 20
        reasons.append("Capped at 20 because stairs-only is a barrier regardless of the extras.")
    return int(np.clip(points, 0, 100)), reasons


# ============ PART D: reference data (your files) ============
def clean(name):
    return " ".join(str(name).split())      # "Baker  Hall" -> "Baker Hall"


@st.cache_data
def load_reference():
    coords = pd.read_csv(HERE / "building_coords.csv").dropna(subset=["lat", "lon"])
    labels = pd.read_csv(HERE / "labels.csv")
    info_path = HERE / "cornell_info.csv"
    info = pd.read_csv(info_path).fillna("unknown") if info_path.exists() \
        else pd.DataFrame(columns=["building"])
    for df in (coords, labels, info):
        df["building"] = df["building"].map(clean)

    # Add every other campus building from OpenStreetMap (all_buildings.csv), skipping
    # ones you already have under a slightly different name (e.g. "Bartels" vs "Bartels Hall").
    all_path = HERE / "all_buildings.csv"
    if all_path.exists():
        extra = pd.read_csv(all_path).dropna(subset=["lat", "lon"])
        extra["building"] = extra["building"].map(clean)
               # Drop things that aren't Cornell campus buildings:
        # street addresses, apartments, frat/sorority houses, acronyms and off-campus businesses
        street = r"\b(?:apartments?|apts?|ave|avenue|street|st|lane|ln|road|rd|place|pl|terrace|way|court|ct)\b"
        greek = (r"\b(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|"
                 r"pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|fraternity|sorority)\b|[\u0370-\u03FF]")
        food = (r"burger|mcdonald|subway|starbucks|dunkin|domino|pizza|taco|wendy|chipotle|kfc|popeyes|"
                r"chick-fil|panera|jimmy john|five guys|insomnia|bagel|wings|sushi|deli\b|grill|"
                r"restaurant|\bbar\b|bank|cvs|pharmacy|7-eleven|mobil|church")
        names = extra["building"]
        not_campus = (names.str.match(r"^\d")                                   # starts with a number
                      | names.str.contains(street, case=False, regex=True)
                      | names.str.contains(greek, case=False, regex=True)
                      | names.str.contains(food, case=False, regex=True)
                      | (names == names.str.upper()))                          # ALL CAPS like "CTB"
        extra = extra[~not_campus]
        mine = [n.lower() for n in coords["building"]]
        is_new = extra["building"].map(lambda n: not any(m in n.lower() or n.lower() in m for m in mine))
        coords = pd.concat([coords, extra[is_new]], ignore_index=True).drop_duplicates("building")
    return coords, labels, info


def label_row_to_features(row):
    # labels.csv says yes/no and wide/narrow -> 1/0
    return {label: int(str(row[label]).strip().lower() in ("yes", "wide")) for label in LABELS}


# ============ PART E: the shared scan database (Google Sheet, "scans" tab) ============
SHEET_COLUMNS = ["timestamp", "building", "entrance", *LABELS, "score"]


@st.cache_resource
def get_sheet():
    client = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
    return client.open_by_url(st.secrets["sheet_url"]).worksheet("scans")


@st.cache_data(ttl=30)    # re-read the sheet at most every 30 seconds
def load_scans():
    return pd.DataFrame(get_sheet().get_all_records(), columns=SHEET_COLUMNS)


def save_scan(building, entrance, features, score):
    row = [datetime.now().isoformat(timespec="seconds"), building, entrance,
           *[features[label] for label in LABELS], "" if score is None else score]
    get_sheet().append_row(row)
    load_scans.clear()    # so the new pin shows up right away


# ============ PART F: scores, colors and the building list ============
COLORS = {"green": "#2E6B4E", "orange": "#B8862B", "red": "#B4473C", "gray": "#8A8F98"}
TINTS = {"green": "#E6EFE9", "orange": "#F6EEDC", "red": "#F5E3E0", "gray": "#EEEFF1"}
VERDICT = {"green": "Good visible access", "orange": "Partial access",
           "red": "Likely barrier", "gray": "Needs human review"}
FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
             '&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" rel="stylesheet">')


def color_for(score):
    if score is None or pd.isna(score):
        return "gray"
    return "green" if score >= 80 else "orange" if score >= 40 else "red"


def building_rows(coords, labels, scans):
    entries = []
    for _, r in labels.iterrows():                      # your hand-checked labels
        score, _ = accessibility_score(label_row_to_features(r))
        entries.append({"building": r["building"], "entrance": str(r["entrance"]).strip(),
                        "score": score, "verified": True})
    for _, r in scans.iterrows():                       # crowdsourced scans
        s = pd.to_numeric(r["score"], errors="coerce")
        entries.append({"building": clean(r["building"]), "entrance": str(r["entrance"]).strip(),
                        "score": None if pd.isna(s) else int(s), "verified": False})
    rows = []
    for _, b in coords.iterrows():
        mine = [e for e in entries if e["building"] == b["building"]]
        if not mine:
            continue
        known = [e["score"] for e in mine if e["score"] is not None]
        rows.append({"building": b["building"], "lat": b["lat"], "lon": b["lon"],
                     "best": max(known) if known else None,      # one good entrance is what matters
                     "photos": sum(e["verified"] for e in mine),
                     "scans": sum(not e["verified"] for e in mine),
                     "review": any(e["score"] is None for e in mine),
                     "entries": mine})
    return sorted(rows, key=lambda b: b["building"])


def in_band(b, band):
    s = b["best"]
    if band == "80–100":
        return s is not None and s >= 80
    if band == "40–79":
        return s is not None and 40 <= s < 80
    if band == "0–39":
        return s is not None and s < 40
    if band == "Needs review":
        return s is None or b["review"]
    return True


# ============ PART G: the map ============
def popup_html(b, info):
    c = color_for(b["best"])
    best = "?" if b["best"] is None else b["best"]
    items = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-top:1px solid #EEECE8;font-size:13px;">'
        f'<span>{escape(e["entrance"].title())} entrance · {"verified" if e["verified"] else "AI scan"}</span>'
        f'<b style="color:{COLORS[color_for(e["score"])]};">{"?" if e["score"] is None else e["score"]}</b></div>'
        for e in b["entries"])
    html = (f'<div style="font-family:Inter,sans-serif;width:250px;color:#1C1917;">'
            f'<div style="font-size:10px;letter-spacing:.16em;font-weight:700;color:#2E6B4E;">ENTRANCE EVIDENCE</div>'
            f'<div style="font-family:\'Source Serif 4\',Georgia,serif;font-size:22px;margin:6px 0 2px;">{escape(b["building"])}</div>'
            f'<div style="font-size:12px;color:#78716C;margin-bottom:10px;">Cornell campus</div>'
            f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:10px;">'
            f'<span style="font-size:40px;font-weight:700;line-height:1;color:{COLORS[c]};">{best}</span>'
            f'<span style="font-size:12px;color:#57534E;">out of 100<br>best entrance</span></div>{items}')
    match = info[info["building"] == b["building"]]
    if not match.empty:
        i = match.iloc[0]
        html += (f'<div style="margin-top:10px;font-size:12px;color:#44403C;"><b>Cornell records</b><br>'
                 f'Elevator: {escape(str(i.get("elevator", "unknown")))}<br>'
                 f'Accessible restroom: {escape(str(i.get("accessible_restroom", "unknown")))}<br>'
                 f'Preferred entrance: {escape(str(i.get("preferred_entrance", "unknown")))}</div>')
    html += ('<div style="font-size:11px;color:#A8A29E;margin-top:10px;">'
             'Building reference pin, not an exact entrance location.</div></div>')
    return html


def build_map(rows, info, unscanned):
    m = folium.Map(location=[42.4475, -76.4815], zoom_start=16, tiles="OpenStreetMap",
                   control_scale=True)
    m.get_root().header.add_child(folium.Element(FONT_LINK))
    for _, u in unscanned.iterrows():                    # small gray dots: not scanned yet
        folium.CircleMarker([u["lat"], u["lon"]], radius=5, color="#78716C", weight=1.5,
                            fill=True, fill_color="#FFFFFF", fill_opacity=0.9,
                            tooltip=f'{u["building"]} · not scanned yet').add_to(m)
    for b in rows:
        c = color_for(b["best"])
        label = "?" if b["best"] is None else b["best"]
        icon = folium.DivIcon(
            html=(f'<div style="box-sizing:border-box;width:36px;height:36px;border-radius:50%;'
                  f'background:{COLORS[c]};border:3px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.35);'
                  f'color:#fff;font:700 12px Inter,sans-serif;display:flex;align-items:center;'
                  f'justify-content:center;">{label}</div>'),
            icon_size=(36, 36), icon_anchor=(18, 18))
        folium.Marker([b["lat"], b["lon"]], icon=icon, tooltip=b["building"],
                      popup=folium.Popup(popup_html(b, info), max_width=300)).add_to(m)
    return m


# ============ PART H: look and feel ============
st.markdown(f"""{FONT_LINK}<style>
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{display: none !important;}}
.block-container {{padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1400px;}}
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li {{font-family: 'Inter', sans-serif;}}
.topbar {{display: flex; align-items: center; justify-content: space-between; padding: 6px 0 16px;
          border-bottom: 1px solid #E7E5E4; margin-bottom: 4px;}}
.brand {{display: flex; align-items: center; gap: 12px; font-size: 1.6rem; font-weight: 700; color: #1C1917;
         letter-spacing: -0.02em;}}
.logo {{width: 44px; height: 44px; border-radius: 12px; background: #2E6B4E; display: grid; place-items: center;}}
.place {{color: #57534E; font-size: 0.95rem;}}
.live {{display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #2E6B4E;
        margin-right: 8px; vertical-align: 1px;}}
.eyebrow {{font-size: 0.72rem; letter-spacing: 0.16em; font-weight: 700; color: #57534E;
           text-transform: uppercase; margin: 14px 0 6px;}}
.headline {{font-family: 'Source Serif 4', Georgia, serif; font-size: 2.6rem; line-height: 1.1;
            color: #1C1917; margin: 0 0 10px; font-weight: 500; letter-spacing: -0.01em;}}
.lede {{color: #57534E; font-size: 1rem; line-height: 1.55; margin-bottom: 18px;}}
.stats {{display: flex; gap: 36px; margin-bottom: 16px;}}
.stats b {{display: block; font-size: 2rem; font-weight: 700; color: #1C1917; line-height: 1.1;}}
.stats span {{font-size: 0.85rem; color: #57534E;}}
.list-head {{display: flex; justify-content: space-between; font-size: 0.85rem; color: #57534E;
             padding: 10px 2px; border-top: 1px solid #E7E5E4; border-bottom: 1px solid #E7E5E4;}}
.row {{display: flex; align-items: center; gap: 14px; padding: 12px 4px; border-bottom: 1px solid #F0EEEA;}}
.badge {{width: 46px; height: 46px; border-radius: 50%; display: grid; place-items: center;
         font-weight: 700; font-size: 1rem; flex-shrink: 0;}}
.row-name {{font-weight: 600; color: #1C1917;}}
.row-meta {{font-size: 0.85rem; color: #78716C;}}
.legend {{display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.88rem; color: #44403C; margin-top: 10px;}}
.dot {{display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; vertical-align: -1px;}}
.foot {{color: #78716C; font-size: 0.82rem; margin-top: 10px;}}
.evidence {{background: #fff; border: 1px solid #E7E5E4; border-radius: 16px; padding: 24px;
            box-shadow: 0 8px 24px rgba(28,25,23,0.06); margin-bottom: 14px;}}
.ev-eyebrow {{font-size: 0.7rem; letter-spacing: 0.16em; font-weight: 700; color: #2E6B4E;}}
.ev-title {{font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; color: #1C1917; margin: 6px 0 2px;}}
.ev-sub {{color: #78716C; font-size: 0.9rem; padding-bottom: 16px; border-bottom: 1px solid #EEECE8;}}
.ev-score {{display: flex; align-items: center; gap: 14px; padding: 16px 0;}}
.big {{font-size: 3.4rem; font-weight: 700; line-height: 1;}}
.ev-out {{font-size: 0.9rem; color: #57534E;}}
.check {{display: flex; gap: 12px; align-items: baseline; padding: 7px 0; font-size: 1rem; color: #1C1917;}}
.ok {{color: #2E6B4E; font-weight: 700;}}
.miss {{color: #B8862B; font-weight: 700;}}
.bad {{color: #B4473C; font-weight: 700;}}
.tag {{font-size: 0.72rem; background: #F6EEDC; color: #8A5A0B; border-radius: 999px; padding: 2px 8px;
       margin-left: 6px; font-weight: 600;}}
.ev-notes {{background: #FAFAF7; border-radius: 10px; padding: 12px 14px; margin-top: 12px;
            font-size: 0.82rem; color: #78716C; line-height: 1.5;}}
.placeholder {{border: 1.5px dashed #D6D3D1; border-radius: 16px; padding: 48px 24px; text-align: center;
               color: #78716C;}}
.stTabs [data-baseweb="tab"] {{font-size: 1rem; font-weight: 600;}}
.stButton > button {{width: 100%; border-radius: 12px; padding: 0.75rem; font-weight: 700; font-size: 1rem;}}
[data-testid="stImage"] img {{border-radius: 14px;}}
</style>""", unsafe_allow_html=True)

LOGO = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 8V5a2 2 0 0 1 2-2h3"/><path d="M16 3h3a2 2 0 0 1 2 2v3"/>'
        '<path d="M21 16v3a2 2 0 0 1-2 2h-3"/><path d="M8 21H5a2 2 0 0 1-2-2v-3"/>'
        '<path d="M9 17v-5a3 3 0 0 1 6 0v5"/><path d="M7.5 17h9"/></svg>')


def list_html(rows):
    out = []
    for b in rows:
        c = color_for(b["best"])
        label = "?" if b["best"] is None else b["best"]
        bits = [f'{b["photos"]} photo' + ("" if b["photos"] == 1 else "s")]
        if b["scans"]:
            bits.append(f'{b["scans"]} community scan' + ("" if b["scans"] == 1 else "s"))
        if b["review"]:
            bits.append("review needed")
        out.append(f'<div class="row"><div class="badge" style="background:{TINTS[c]};color:{COLORS[c]};">{label}</div>'
                   f'<div><div class="row-name">{escape(b["building"])}</div>'
                   f'<div class="row-meta">{" · ".join(bits)}</div></div></div>')
    return "".join(out) or '<p class="foot">No buildings match.</p>'


def evidence_card(building, entrance, features, scores, score, reasons):
    c = color_for(score)
    number = "?" if score is None else score
    checks = [("ramp", "Ramp", "No ramp", True),
              ("stairs_only", "Stairs only: a barrier", "Not stairs-only", False),
              ("level_entrance", "Level entrance", "No level entrance", True),
              ("door_width", "Wide door", "Narrow door", True),
              ("auto_opener", "Auto-opener", "No auto-opener", True)]
    items = ""
    for key, yes_text, no_text, good_when_yes in checks:
        present = bool(features[key])
        text = yes_text if present else no_text
        if key == "stairs_only" and present:
            mark = '<span class="bad">✗</span>'
        elif present == good_when_yes:
            mark = '<span class="ok">✓</span>'
        else:
            mark = '<span class="miss">○</span>'
        unsure = '<span class="tag">unsure</span>' if 0.35 <= scores[key] <= 0.65 else ""
        items += f'<div class="check">{mark}<span>{text}{unsure}</span></div>'
    return (f'<div class="evidence"><div class="ev-eyebrow">ENTRANCE EVIDENCE</div>'
            f'<div class="ev-title">{escape(building)}</div>'
            f'<div class="ev-sub">{entrance.title()} entrance · Cornell campus</div>'
            f'<div class="ev-score"><span class="big" style="color:{COLORS[c]};">{number}</span>'
            f'<span class="ev-out"><b>out of 100</b><br>{VERDICT[c]}</span></div>'
            f'<div>{items}</div>'
            f'<div class="ev-notes">{escape(" ".join(reasons))}<br>Model prediction · verify on site.</div></div>')


# ============ PART I: the screens ============
coords, labels, info = load_reference()
try:
    scans = load_scans()
except Exception:
    scans = pd.DataFrame(columns=SHEET_COLUMNS)
rows = building_rows(coords, labels, scans)

st.markdown(f'<div class="topbar"><div class="brand"><div class="logo">{LOGO}</div>AccessScan</div>'
            f'<div class="place"><span class="live"></span>Cornell University · Ithaca, NY</div></div>',
            unsafe_allow_html=True)

explore_tab, scan_tab, about_tab = st.tabs(["Explore map", "Scan an entrance", "About"])

with explore_tab:
    left, right = st.columns([1, 2.1], gap="large")
    with left:
        st.markdown('<div class="eyebrow">Campus entrance explorer</div>'
                    '<div class="headline">A clearer way in.</div>'
                    '<div class="lede">Find the best way into a building before you get there. '
                    'Tap a pin to see each entrance\'s ramp, level entry, door width and auto-opener.</div>'
                    f'<div class="stats"><div><b>{len(labels)}</b><span>photo records</span></div>'
                    f'<div><b>{len(rows)}</b><span>buildings scanned</span></div>'
                    f'<div><b>{len(scans)}</b><span>community scans</span></div></div>',
                    unsafe_allow_html=True)
        query = st.text_input("Search", placeholder="Search a building…", label_visibility="collapsed")
        band = st.pills("Filter", ["All", "80–100", "40–79", "0–39", "Needs review"],
                        default="All", label_visibility="collapsed")
        shown = [b for b in rows if in_band(b, band or "All") and query.strip().lower() in b["building"].lower()]
        st.markdown(f'<div class="list-head"><span>{len(shown)} buildings</span>'
                    f'<span>Best entrance score</span></div>', unsafe_allow_html=True)
        with st.container(height=360, border=False):
            st.markdown(list_html(shown), unsafe_allow_html=True)
        st.markdown('<p class="foot">A photo cannot verify a safe route or ADA compliance.</p>',
                    unsafe_allow_html=True)
    with right:
        scanned_names = {b["building"] for b in rows}
        unscanned = coords[~coords["building"].isin(scanned_names)]
        unscanned = unscanned[unscanned["building"].str.lower().str.contains(query.strip().lower(), regex=False)]
        if (band or "All") != "All":
            unscanned = unscanned.iloc[0:0]          # only show gray dots under "All"
        st_folium(build_map(shown, info, unscanned), height=660, use_container_width=True,
                  returned_objects=[], key="explore_map")
        st.markdown('<div class="legend"><b>Photo-based entrance score / 100</b>'
                    f'<span><span class="dot" style="background:{COLORS["green"]}"></span>80–100</span>'
                    f'<span><span class="dot" style="background:{COLORS["orange"]}"></span>40–79</span>'
                    f'<span><span class="dot" style="background:{COLORS["red"]}"></span>0–39</span>'
                    f'<span><span class="dot" style="background:{COLORS["gray"]}"></span>Review</span>'
                    '<span><span class="dot" style="background:#fff;border:1.5px solid #78716C;"></span>'
                    'Not scanned yet</span></div>',
                    unsafe_allow_html=True)

with scan_tab:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown('<div class="eyebrow">Scan an entrance</div>'
                    '<div class="headline">Help the next student.</div>'
                    '<div class="lede">Photograph a building entrance. The model checks five visible '
                    'features, and your result joins the campus map.</div>', unsafe_allow_html=True)
        building = st.selectbox("Building", sorted(coords["building"]))
        entrance = st.radio("Entrance", ["front", "back"], horizontal=True)
        method = st.radio("Photo", ["Upload a photo", "Take a photo"], horizontal=True)
        if method == "Take a photo":
            photo = st.camera_input("Point your camera at the entrance (spaces, not people)")
        else:
            photo = st.file_uploader("Photo of the entrance", type=["jpg", "jpeg", "png"],
                                     help="Photograph the space, not people.")
        image = ImageOps.exif_transpose(Image.open(photo)) if photo else None
        if image is not None:
            st.image(image)
    with right:
        if image is None:
            st.markdown('<div class="placeholder"><div class="ev-eyebrow">ENTRANCE EVIDENCE</div>'
                        '<p style="margin-top:10px;">Your result will appear here after you add a photo.</p></div>',
                        unsafe_allow_html=True)
        else:
            features, scores = predict(image)
            score, reasons = accessibility_score(features)
            st.markdown(evidence_card(building, entrance, features, scores, score, reasons),
                        unsafe_allow_html=True)
            if st.button("Add this entrance to the campus map", type="primary"):
                try:
                    save_scan(building, entrance, features, score)
                    st.success("Added! Open the Explore map tab to see it.")
                except Exception as e:
                    st.error(f"Couldn't save the scan: {e}")

with about_tab:
    st.markdown(
        '<div class="eyebrow">About</div><div class="headline">About AccessScan</div>'
        '<div class="lede" style="max-width:760px;">'
        '<b>The problem.</b> For students who use wheelchairs, crutches, or other mobility aids, finding an '
        'accessible way into a campus building often means arriving in person and searching for a ramp, a level '
        'entrance, or a door with an automatic opener. Official accessibility maps exist, but they are difficult '
        'to find and do not reflect current conditions such as a broken door opener or a blocked path.<br><br>'
        '<b>How it works.</b> AccessScan allows any student to photograph a building entrance and receive an '
        'accessibility score within seconds. A MobileNetV3 image classifier, fine-tuned on labeled photographs of '
        'Cornell entrances, detects five visible features: a ramp, stairs without an alternative route, a level '
        'entrance, a wide door, and an automatic door opener. A transparent scoring rule converts these features '
        'into a score from 0 to 100 and explains how each point was earned, and every scan is added to a shared '
        'campus map so that each contribution benefits the students who come after.<br><br>'
        '<b>Privacy.</b> AccessScan is designed to photograph spaces rather than people. Uploaded images are used '
        'only to generate a score and are never stored; only the resulting feature predictions and score are saved.'
        '</div>',
        unsafe_allow_html=True)