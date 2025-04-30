# preprocessing.py
import pandas as pd
import re

# === FONCTIONS UTILITAIRES ===
def extract_numeric_prefix(value):
    if isinstance(value, str) and "-" in value:
        match = re.match(r"(\d{2})", value)
        if match:
            return int(match.group(1))
    return value

def extract_numeric_comparator(value):
    if isinstance(value, str) and (">=" in value or "<=" in value):
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return value

def extract_numeric_comparator2(value):
    if isinstance(value, str) and ">" in value:
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return value

def extract_prefix_code(value, prefix, pos):
    if isinstance(value, str) and value.startswith(prefix):
        return value[pos]
    return value

def replace_values(value):
    return {"O": 1, "N": 0, "R": 2}.get(value, value)

def replace_values_caract4(value):
    mapping = {
        "absence de surface": 0,
        "Surface de moins d": 1,
        "Surface entre 501": 2,
        "Surface entre 1001": 3,
        "Surface entre 1501": 4,
        "Surface de plus de": 5,
    }
    return mapping.get(value, value)

def replace_values_plus7000(value):
    return 8000 if value == "7000+" else value

# === TRAITEMENT PRINCIPAL ===
def clean_dataframe(df):
    drop_columns = [
        "FRCH1", "FRCH2", "DEROG12", "DEROG13", "DEROG14", "RISK6", "RISK8", "RISK9",
        "RISK12", "RISK13", "EQUIPEMENT2", "EQUIPEMENT5", "ESPINSEE", "TYPBAT1", "DEROG16"
    ]
    df_clean = df.drop(columns=drop_columns, errors='ignore')

    # Apply generic extract functions
    for func in [extract_numeric_prefix, extract_numeric_comparator, extract_numeric_comparator2]:
        df_clean = df_clean.applymap(func)

    # Apply specific column code extraction
    if "ACTIVIT2" in df_clean.columns:
        df_clean["ACTIVIT2"] = df_clean["ACTIVIT2"].apply(lambda x: extract_prefix_code(x, "ACT", 3))
    if "VOCATION" in df_clean.columns:
        df_clean["VOCATION"] = df_clean["VOCATION"].apply(lambda x: extract_prefix_code(x, "VOC", 3))
    if "ANCIENNETE" in df_clean.columns:
        df_clean["ANCIENNETE"] = df_clean["ANCIENNETE"].apply(lambda x: extract_prefix_code(x, "CLASS", 4))
    if "AN_EXERC" in df_clean.columns:
        df_clean["AN_EXERC"] = df_clean["AN_EXERC"].apply(lambda x: extract_prefix_code(x, "ANNEE", 5))
    if "INDEM2" in df_clean.columns:
        df_clean["INDEM2"] = df_clean["INDEM2"].apply(lambda x: extract_prefix_code(x, "CLASS", 5))

    # Apply replacements
    df_clean = df_clean.applymap(replace_values)
    if "CARACT4" in df_clean.columns:
        df_clean["CARACT4"] = df_clean["CARACT4"].apply(replace_values_caract4)
    for col in ["SURFACE4", "SURFACE6"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(replace_values_plus7000)

    return df_clean

# === SCRIPT EXÉCUTABLE ===
if __name__ == "__main__":
    df = pd.read_csv("test_input.csv", low_memory=False)
    df_clean = clean_dataframe(df)
    df_clean.to_csv("test_input_cleaned.csv", index=False)
    print("✅ Nettoyage terminé et exporté sous 'test_input_cleaned.csv'")
