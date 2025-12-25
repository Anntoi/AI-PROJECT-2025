import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Définition des chemins
RAW_DATA_DIR = "/home/kmg/Téléchargements/Dogs Vs AiDogs_CUTTED/data/raw"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


def load_labels_from_txt(labels_dir, image_filename):
    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_filename)

    if not os.path.exists(label_path):
        print(f"Avertissement: Label manquant pour {image_filename}")
        return None

    try:
        with open(label_path, 'r') as f:
            label = int(f.readline().strip())
            return label
    except:
        print(f"Erreur de lecture du label pour {image_filename}")
        return None


def create_generators(raw_data_dir=RAW_DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    print("--- Préparation des Générateurs de Données ---")

    datasets = {
        "train": ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        ),
        "valid": ImageDataGenerator(rescale=1./255),
        "test": ImageDataGenerator(rescale=1./255)
    }

    generators = {}

    for set_name, datagen in datasets.items():
        set_path = os.path.join(raw_data_dir, set_name)
        images_dir = os.path.join(set_path, "images")
        labels_dir = os.path.join(set_path, "labels")

        # Vérification existence dossiers
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"❌ Erreur : Dossiers {set_name}/images ou {set_name}/labels introuvables dans : {raw_data_dir}")
            return None, None, None

        # Construction du dataframe
        files = os.listdir(images_dir)
        data = []

        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                label = load_labels_from_txt(labels_dir, filename)
                if label is not None:
                    class_label = "IA" if label == 1 else "Reel"
                    data.append({"filename": filename, "class": class_label})

        if len(data) == 0:
            print(f" Aucun fichier valide dans {set_name}")
            continue

        df = pd.DataFrame(data)
        print(f"{set_name}: {len(df)} images trouvées.")

        gen = datagen.flow_from_dataframe(
            dataframe=df,
            directory=images_dir,
            x_col="filename",
            y_col="class",
            target_size=image_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=(set_name == "train")
        )

        generators[set_name] = gen

    return generators.get("train"), generators.get("valid"), generators.get("test")


if __name__ == "__main__":
    train_gen, val_gen, test_gen = create_generators()
    print("Générateurs créés.")

