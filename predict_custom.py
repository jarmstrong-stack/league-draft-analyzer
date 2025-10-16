import torch, yaml, json
from source.LDANet import LDANet        # ✅ correct import
from source.Normalizer import Normalizer
import source.constants as CONST

# Load champion mapping
with open("data/champ_mapping.yml", "r", encoding="utf-8") as f:
    champ_to_int = yaml.safe_load(f)
name_map = {k.lower().replace(" ", "").replace("'", ""): v for k, v in champ_to_int.items()}

def champ_to_id(name):
    key = name.lower().replace(" ", "").replace("'", "")
    return name_map.get(key)

def input_team(side):
    champs = []
    print(f"\nEnter 5 champions for {side} side:")
    while len(champs) < 5:
        champ = input(f"  {len(champs)+1}. ").strip()
        cid = champ_to_id(champ)
        if cid is None:
            print("  ⚠️  Not found in mapping, try again.")
        else:
            champs.append(cid)
    return champs

def main():
    blue = input_team("BLUE")
    red = input_team("RED")

    game = {
        "pick": {
            "blue": {str(i+1): blue[i] for i in range(5)},
            "red": {str(i+1): red[i] for i in range(5)},
        },
        "ban": {"blue": [], "red": []},
        "patch": 1510,  # current patch; not critical
    }

    print("\n🔹 Loading model...")
    model = LDANet()
    model.load_model("data/lda_net.pth")
    model.eval()

    norm = Normalizer(features_to_process=["pick", "ban", "patch"])
    normalized = norm.normalize(game)

    with torch.no_grad():
        output = model(normalized)
        prob_blue_win = float(torch.sigmoid(output).item())

    print(f"\n✅ Predicted BLUE win probability: {prob_blue_win:.3f}")
    print(f"   → RED win probability: {1 - prob_blue_win:.3f}")

if __name__ == "__main__":
    main()
