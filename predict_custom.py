# predict_custom.py — type champs -> get win chance (compatible with this repo)
import os, sys, json, yaml, torch

# --- Make ./source importable from repo root ---
HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "source")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- Preload 'train' so source/LDANet.py's "import train" works ---
import importlib.util
_train_spec = importlib.util.spec_from_file_location("train", os.path.join(SRC, "train.py"))
_train_mod  = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)
sys.modules["train"] = _train_mod

# --- Now import repo modules as LDANet.py expects ---
from LDANet import LDANet
import constants as CONST

CHAMP_MAP_PATH = os.path.join("data", "champ_mapping.yml")
MODEL_PATH     = CONST.LDA_WEIGHTS_PATH  # "data/lda_net.pth" per constants.py

def _load_mapping(path=CHAMP_MAP_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # normalize keys to tolerate spaces, apostrophes, and dots (K'Sante, Kai'Sa, Dr. Mundo)
    norm = {}
    for k, v in raw.items():
        key = k.lower().replace(" ", "").replace("'", "").replace(".", "")
        norm[key] = v
    return norm

def _champ_to_id(name: str, norm_map: dict):
    key = name.strip().lower().replace(" ", "").replace("'", "").replace(".", "")
    return norm_map.get(key)

def _input_team(side: str, norm_map: dict):
    champs = []
    print(f"\nEnter 5 champions for {side} side:")
    while len(champs) < 5:
        user = input(f"  {len(champs)+1}. ").strip()
        cid = _champ_to_id(user, norm_map)
        if cid is None:
            print("  ⚠️  Not found. Check spelling (e.g., K'Sante, Kai'Sa, Dr. Mundo). Try again.")
            continue
        champs.append(cid)
    return champs

def _build_game(blue_ids, red_ids, patch_int=1510):
    return {
        "pick": {
            "blue": {str(i+1): blue_ids[i] for i in range(5)},
            "red":  {str(i+1): red_ids[i]  for i in range(5)},
        },
        "ban": {"blue": [], "red": []},  # optional
        "patch": patch_int,              # not critical for inference
    }

def main():
    # 1) Sanity checks
    if not os.path.exists(CHAMP_MAP_PATH):
        print(f"❌ Missing champion mapping: {CHAMP_MAP_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Missing model file: {MODEL_PATH} (train first to create it)")
        return

    # 2) Collect teams
    norm_map = _load_mapping()
    blue = _input_team("BLUE", norm_map)
    red  = _input_team("RED",  norm_map)
    game = _build_game(blue, red)

    # 3) Build model, load weights
    print("\n🔹 Loading model…")
    net = LDANet().to(CONST.DEVICE_CUDA)
    net.load_lda(MODEL_PATH)          # <-- correct loader for this repo
    net.eval()

    # 4) Use the repo’s built-in preprocessing helper
    #    handle_prediction_data() will compute synergies if missing, translate names if needed,
    #    and normalize before forward pass.
    features = net.handle_prediction_data(game)

    with torch.no_grad():
        logits = net(features)                # shape [1], raw logit
        p_blue = float(torch.sigmoid(logits).item())

    print(f"\n✅ Predicted BLUE win probability: {p_blue:.3f}")
    print(f"   → RED win probability: {1.0 - p_blue:.3f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error during prediction:", repr(e))
