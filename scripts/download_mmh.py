from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/skishore/makemeahanzi/master"
FILES = ("graphics.txt", "dictionary.txt")

out = Path("data")
out.mkdir(exist_ok=True)
for name in FILES:
    target = out / name
    print(f"Downloading {name} -> {target}")
    urlretrieve(f"{BASE}/{name}", target)
print("Done.")
