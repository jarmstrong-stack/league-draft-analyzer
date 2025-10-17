# predict_gui.py — minimal GUI for custom draft predictions
# Works with your current repo layout (loads source/*, uses data/lda_net.pth)

import os, sys, warnings, json, yaml, torch, traceback
warnings.filterwarnings("ignore", category=UserWarning)

# --- Make ./source importable ---
HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "source")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- Preload 'train' so source/LDANet.py's "import train" resolves ---
import importlib.util
_train_spec = importlib.util.spec_from_file_location("train", os.path.join(SRC, "train.py"))
_train_mod  = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)
sys.modules["train"] = _train_mod

# --- Repo modules ---
from LDANet import LDANet
import constants as CONST

# --- GUI ---
import tkinter as tk
from tkinter import ttk, messagebox

CHAMP_MAP_PATH = os.path.join("data", "champ_mapping.yml")
MODEL_PATH     = getattr(CONST, "LDA_WEIGHTS_PATH", os.path.join("data","lda_net.pth"))

def load_mapping():
    if not os.path.exists(CHAMP_MAP_PATH):
        raise FileNotFoundError(f"Missing champion mapping: {CHAMP_MAP_PATH}")
    with open(CHAMP_MAP_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # exact names for dropdown display
    names = sorted(raw.keys(), key=str.lower)

    # normalized lookup (handles K'Sante, Kai'Sa, Dr. Mundo typing)
    norm = {}
    for k, v in raw.items():
        key = k.lower().replace(" ", "").replace("'", "").replace(".", "")
        norm[key] = v
    return raw, names, norm

RAW_MAP, CHAMP_NAMES, NORM_MAP = load_mapping()

def name_to_id(name: str):
    key = (name or "").strip().lower().replace(" ", "").replace("'", "").replace(".", "")
    return NORM_MAP.get(key)

def build_game(blue_ids, red_ids, patch_int=1510):
    return {
        "pick": {
            "blue": {str(i+1): blue_ids[i] for i in range(5)},
            "red":  {str(i+1): red_ids[i]  for i in range(5)},
        },
        "ban": {"blue": [], "red": []},  # optional
        "patch": patch_int,
    }

def safe_make_model():
    # Try device from constants; if it fails (no CUDA), retry on CPU
    try:
        net = LDANet().to(CONST.DEVICE_CUDA)
        net.load_lda(MODEL_PATH)
        net.eval()
        return net
    except Exception as e:
        # Retry on CPU if CUDA failed
        try:
            net = LDANet().to("cpu")
            net.load_lda(MODEL_PATH)
            net.eval()
            return net
        except Exception:
            raise e

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("League Draft Analyzer — Quick Predict")
        self.geometry("780x420")
        self.minsize(760, 420)

        # Title
        title = ttk.Label(self, text="Custom Draft Prediction", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(12, 6))

        # Frame for pickers
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=12)

        left = ttk.LabelFrame(frame, text="BLUE side (5 champs)")
        left.grid(row=0, column=0, padx=(0,8), sticky="nsew")
        right = ttk.LabelFrame(frame, text="RED side (5 champs)")
        right.grid(row=0, column=1, padx=(8,0), sticky="nsew")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Comboboxes
        self.blue_boxes = []
        self.red_boxes  = []
        for i in range(5):
            cb = ttk.Combobox(left, values=CHAMP_NAMES, state="readonly")
            cb.grid(row=i, column=0, padx=8, pady=6, sticky="ew")
            self.blue_boxes.append(cb)

            cb2 = ttk.Combobox(right, values=CHAMP_NAMES, state="readonly")
            cb2.grid(row=i, column=0, padx=8, pady=6, sticky="ew")
            self.red_boxes.append(cb2)

        # Buttons row
        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=12, pady=(10,4))

        self.predict_btn = ttk.Button(btns, text="Predict", command=self.on_predict)
        self.predict_btn.pack(side="left")

        swap_btn = ttk.Button(btns, text="Swap Sides", command=self.on_swap)
        swap_btn.pack(side="left", padx=8)

        clear_btn = ttk.Button(btns, text="Clear", command=self.on_clear)
        clear_btn.pack(side="left")

        # Result
        self.result_var = tk.StringVar(value="Pick champs then click Predict.")
        res = ttk.Label(self, textvariable=self.result_var, font=("Segoe UI", 12))
        res.pack(padx=12, pady=10, anchor="w")

        # Status
        self.status_var = tk.StringVar(value=f"Model: {os.path.basename(MODEL_PATH)}  |  Device pref: {getattr(CONST, 'DEVICE_CUDA','cpu')}")
        status = ttk.Label(self, textvariable=self.status_var, foreground="#666")
        status.pack(padx=12, pady=(0,10), anchor="w")

    def on_swap(self):
        blue_vals = [cb.get() for cb in self.blue_boxes]
        red_vals  = [cb.get() for cb in self.red_boxes]
        for i in range(5):
            self.blue_boxes[i].set(red_vals[i] if red_vals[i] else "")
            self.red_boxes[i].set(blue_vals[i] if blue_vals[i] else "")
        self.result_var.set("Sides swapped. Click Predict.")

    def on_clear(self):
        for cb in self.blue_boxes + self.red_boxes:
            cb.set("")
        self.result_var.set("Cleared. Pick champs then click Predict.")

    def on_predict(self):
        try:
            blue_names = [cb.get() for cb in self.blue_boxes]
            red_names  = [cb.get() for cb in self.red_boxes]

            if any(not n for n in blue_names+red_names):
                messagebox.showwarning("Missing champs", "Please select 5 champs for BLUE and 5 for RED.")
                return

            # map names -> ids
            blue_ids = []
            red_ids  = []
            for n in blue_names:
                cid = name_to_id(n)
                if cid is None:
                    messagebox.showerror("Unknown champion", f"'{n}' not in mapping. Check spelling.")
                    return
                blue_ids.append(cid)
            for n in red_names:
                cid = name_to_id(n)
                if cid is None:
                    messagebox.showerror("Unknown champion", f"'{n}' not in mapping. Check spelling.")
                    return
                red_ids.append(cid)

            game = build_game(blue_ids, red_ids)

            self.result_var.set("Loading model…")
            self.update_idletasks()

            net = safe_make_model()

            # Use repo's helper to do normalization/features
            feats = net.handle_prediction_data(game)

            with torch.no_grad():
                logit = net(feats)
                p_blue = float(torch.sigmoid(logit).item())

            self.result_var.set(f"BLUE win prob: {p_blue:.3f}   |   RED: {1.0 - p_blue:.3f}")

        except Exception as e:
            tb = traceback.format_exc(limit=2)
            self.result_var.set("Error during prediction. See dialog.")
            messagebox.showerror("Error", f"{e}\n\n{tb}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
