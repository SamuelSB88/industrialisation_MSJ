# 🧹 Script de Nettoyage des Données — CM Prediction

Ce dépôt contient un script Python modulaire pour nettoyer un jeu de données en entrée (`test_input.csv`) avant de l'utiliser pour la prédiction du coût moyen (CM).

---

## 📁 Contenu

- `preprocessing.py` : script principal avec toutes les fonctions de nettoyage factorisées
- `test_input.csv` : (à fournir) fichier brut à nettoyer
- `test_input_cleaned.csv` : fichier nettoyé généré automatiquement

---

## ⚙️ Utilisation

1. Placez `test_input.csv` dans le même dossier que `preprocessing.py`
2. Lancez ce script :

```bash
python preprocessing.py
```

3. Le script produit automatiquement un fichier propre :
```
test_input_cleaned.csv
```

---

## 🧪 Fonctions de nettoyage incluses

- Suppression de colonnes à types mixtes
- Extraction de préfixes numériques
- Conversion des chaînes `ACT`, `VOC`, `CLASS`, etc.
- Remplacement des valeurs `"O"`, `"N"`, `"R"` par `1`, `0`, `2`
- Codage ordinal pour certaines variables catégorielles

---

## ✅ Dépendances

- `pandas`
- `re` (standard Python)

---

## 🧾 Auteur

Projet préparé pour le nettoyage de données pré-modélisation CM — à intégrer dans pipeline de machine learning.

