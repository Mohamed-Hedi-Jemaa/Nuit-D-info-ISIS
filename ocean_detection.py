import os
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET_PATH = "dataset"  # Le dossier où vous extrairez le dataset téléchargé manuellement
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")  # Le fichier de configuration du dataset

# Train YOLOv8 model
try:
    # Vérifier que le dataset existe
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Le dossier dataset '{DATASET_PATH}' n'existe pas. "
            "Veuillez télécharger le dataset depuis Roboflow et l'extraire dans ce dossier."
        )

    # Vérifier que data.yaml existe
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Le fichier '{DATA_YAML}' n'existe pas. "
            "Assurez-vous que le dataset a été correctement extrait."
        )
    logger.info(f"Dataset trouvé dans {DATASET_PATH}")
    logger.info(f"Fichier de configuration trouvé : {DATA_YAML}")

    # Initialize model
    logger.info("Initialisation du modèle YOLOv8")
    model = YOLO('yolov8n.pt')  # Charge le modèle pré-entraîné
    logger.info("Modèle initialisé avec succès")

    # Train the model
    logger.info("Démarrage de l'entraînement")
    results = model.train(
        data=DATA_YAML,
        epochs=5,
        imgsz=640,
        batch=8,
        name='yolov8n_custom'
    )
    logger.info("Entraînement terminé avec succès")
    
    # Validation du modèle
    logger.info("Validation du modèle")
    val_results = model.val(data=DATA_YAML)
    logger.info(f"Résultats de validation : mAP={val_results.box.map}")

    # Afficher le chemin des résultats
    logger.info(f"Les résultats de l'entraînement sont sauvegardés dans : {model.trainer.save_dir}")

except Exception as e:
    logger.error(f"Erreur : {str(e)}")
    logger.error("Stack trace:", exc_info=True)
    raise

# Fonction pour tester sur une image
def test_image(model_path, image_path, conf=0.25):
    """
    Teste le modèle sur une image
    Args:
        model_path: Chemin vers le modèle entraîné
        image_path: Chemin vers l'image à tester
        conf: Seuil de confiance (default: 0.25)
    """
    try:
        model = YOLO(model_path)
        results = model.predict(source=image_path, conf=conf, save=True)
        logger.info(f"Prédiction terminée. Résultats sauvegardés dans : {model.predictor.save_dir}")
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction sur l'image : {str(e)}")
        raise

# Fonction pour tester sur une vidéo
def test_video(model_path, video_path, conf=0.25):
    """
    Teste le modèle sur une vidéo
    Args:
        model_path: Chemin vers le modèle entraîné
        video_path: Chemin vers la vidéo à tester
        conf: Seuil de confiance (default: 0.25)
    """
    try:
        model = YOLO(model_path)
        results = model.predict(source=video_path, conf=conf, save=True)
        logger.info(f"Prédiction terminée. Résultats sauvegardés dans : {model.predictor.save_dir}")
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction sur la vidéo : {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Chemin vers le meilleur modèle après l'entraînement
        best_model_path = os.path.join("runs", "detect", "yolov8n_custom", "weights", "best.pt")
        
        # Exemple d'utilisation pour une image (décommentez et modifiez les chemins selon vos besoins)
        # test_image(best_model_path, "chemin/vers/votre/image.jpg")
        
        # Exemple d'utilisation pour une vidéo (décommentez et modifiez les chemins selon vos besoins)
        # test_video(best_model_path, "chemin/vers/votre/video.mp4")
        
    except Exception as e:
        logger.error(f"Erreur : {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise
