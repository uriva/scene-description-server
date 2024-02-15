from typing import Dict

import modal

from src import worker

stub = modal.Stub("image-describe-service")


model = None


@stub.cls(
    image=(
        modal.Image.debian_slim()
        .apt_install("git")
        .run_commands(
            [
                "cd home && git clone -b v1.0 https://github.com/camenduru/LLaVA",
                "cd home/LLaVA && pip install .",
            ],
            gpu="any",
        )
        .pip_install("gamla")
        .run_function(worker.load_model, gpu="any")
    ),
    gpu="any",
)
class Model:
    @modal.enter()
    def load_model(self):
        self._model = worker.load_model()

    @modal.web_endpoint(method="POST")
    def predict(self, request: Dict):
        return worker.work_on_file(self._model, request["imageUrl"], request["prompt"])
