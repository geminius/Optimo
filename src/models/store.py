"""Minimal ModelStore stub for testing purposes."""


class ModelStore:
    """Placeholder model store implementation."""

    def save_model(self, model, *args, **kwargs):  # pragma: no cover - stub
        return "model-id"

    def load_model(self, model_id: str):  # pragma: no cover - stub
        return None

    def get_model_metadata(self, model_id: str):  # pragma: no cover - stub
        return None

