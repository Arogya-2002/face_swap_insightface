from dataclasses import dataclass

@dataclass
class ModelInitializationArtifact:
    model_name: str

@dataclass
class SwapperModelArtifact:
    result_image_path: str