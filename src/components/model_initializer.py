from src.entity.face_swap_config import ConfigEntity, ModelInitializerConfig
from src.entity.face_swap_artifact import ModelInitializationArtifact
from src.exceptions import CustomException
from src.logger import logging

import sys
from insightface.app import FaceAnalysis


class ModelInitializer:
    def __init__(self):
        try:
            logging.info("Creating ModelInitializerConfig...")
            self.model_initializer_config = ModelInitializerConfig(config=ConfigEntity())
        except Exception as e:
            logging.error("Failed to initialize ModelInitializerConfig", exc_info=True)
            raise CustomException(e, sys) from e

    def initialize_model(self) -> FaceAnalysis:
        try:
            logging.info("Starting FaceAnalysis model initialization...")

            app = FaceAnalysis(name=self.model_initializer_config.model_name)
            app.prepare(
                ctx_id=self.model_initializer_config.ctx_id,
                det_size=self.model_initializer_config.det_size
            )

            logging.info("FaceAnalysis model initialized successfully.")

            # Create artifact for tracking (if needed later)
            artifact = ModelInitializationArtifact(
                model_name=self.model_initializer_config.model_name
            )
            logging.info(f"ModelInitializationArtifact created: {artifact}")

            return app

        except Exception as e:
            logging.error("Error during model initialization", exc_info=True)
            raise CustomException(e, sys) from e
