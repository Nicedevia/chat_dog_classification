import pandas as pd

# Charger les associations images/sons
train_df = pd.read_csv("data/train_pairs.csv")
test_df = pd.read_csv("data/test_pairs.csv")

# Vérifier les doublons
duplicates = pd.merge(train_df, test_df, on=["image", "audio"], how="inner")

if duplicates.empty:
    print("✅ Les ensembles de train et test sont bien distincts.")
else:
    print(f"⚠️ {len(duplicates)} doublons trouvés entre train et test :")
    print(duplicates)
