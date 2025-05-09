import pandas as pd
import re

class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def extract_numeric_prefix(value):
        if isinstance(value, str) and "-" in value:
            match = re.match(r"(\d{2})", value)
            if match:
                return int(match.group(1))
        return value

    @staticmethod
    def extract_numeric_comparator(value):
        if isinstance(value, str) and (">=" in value or "<=" in value):
            match = re.search(r"(\d+)", value)
            if match:
                return int(match.group(1))
        return value

    @staticmethod
    def extract_numeric_comparator2(value):
        if isinstance(value, str) and ">" in value:
            match = re.search(r"(\d+)", value)
            if match:
                return int(match.group(1))
        return value

    @staticmethod
    def extract_prefix_code(value, prefix, pos):
        if isinstance(value, str) and value.startswith(prefix):
            return value[pos]
        return value

    @staticmethod
    def replace_values(value):
        return {"O": 1, "N": 0, "R": 2}.get(value, value)

    @staticmethod
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

    @staticmethod
    def replace_values_plus7000(value):
        return 8000 if value == "7000+" else value

    def clean_dataframe(self, df):
        drop_columns = [
            "FRCH1", "FRCH2", "DEROG12", "DEROG13", "DEROG14", "RISK6", "RISK8", "RISK9",
            "RISK12", "RISK13", "EQUIPEMENT2", "EQUIPEMENT5", "ESPINSEE", "TYPBAT1", "DEROG16"
        ]
        df_clean = df.drop(columns=drop_columns, errors='ignore')

        for func in [self.extract_numeric_prefix, self.extract_numeric_comparator, self.extract_numeric_comparator2]:
            df_clean = df_clean.applymap(func)

        if "ACTIVIT2" in df_clean.columns:
            df_clean["ACTIVIT2"] = df_clean["ACTIVIT2"].apply(lambda x: self.extract_prefix_code(x, "ACT", 3))
        if "VOCATION" in df_clean.columns:
            df_clean["VOCATION"] = df_clean["VOCATION"].apply(lambda x: self.extract_prefix_code(x, "VOC", 3))
        if "ANCIENNETE" in df_clean.columns:
            df_clean["ANCIENNETE"] = df_clean["ANCIENNETE"].apply(lambda x: self.extract_prefix_code(x, "CLASS", 4))
        if "AN_EXERC" in df_clean.columns:
            df_clean["AN_EXERC"] = df_clean["AN_EXERC"].apply(lambda x: self.extract_prefix_code(x, "ANNEE", 5))
        if "INDEM2" in df_clean.columns:
            df_clean["INDEM2"] = df_clean["INDEM2"].apply(lambda x: self.extract_prefix_code(x, "CLASS", 5))

        df_clean = df_clean.applymap(self.replace_values)
        if "CARACT4" in df_clean.columns:
            df_clean["CARACT4"] = df_clean["CARACT4"].apply(self.replace_values_caract4)
        for col in ["SURFACE4", "SURFACE6"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.replace_values_plus7000)

        return df_clean

# Exemple exécutable
if __name__ == "__main__":
    df = pd.read_csv("test_input.csv", low_memory=False)
    preproc = Preprocessor()
    df_clean = preproc.clean_dataframe(df)
    df_clean.to_csv("test_input_cleaned.csv", index=False)
    print("✅ Nettoyage terminé et exporté sous 'test_input_cleaned.csv'")
