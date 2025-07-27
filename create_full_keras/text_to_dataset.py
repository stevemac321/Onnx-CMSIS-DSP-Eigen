import re
from pathlib import Path
import pandas as pd

def parse_training_file(path="training_strings.txt"):
    texts, labels = [], []
    current_label = None

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Class"):
                match = re.match(r"Class (\d+):", line)
                if match:
                    current_label = int(match.group(1))
            elif line.startswith("-") and current_label is not None:
                text = line[1:].strip()
                texts.append(text)
                labels.append(current_label)

    return pd.DataFrame({"text": texts, "label": labels})

# === Save the parsed dataset to a CSV ===
if __name__ == "__main__":
    df = parse_training_file()
    output_path = Path("data/dataset.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to {output_path.resolve()}")