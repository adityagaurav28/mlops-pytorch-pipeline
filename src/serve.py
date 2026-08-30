import io
import os

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
MODEL_ARCH = os.getenv("MODEL_ARCH", "simple_cnn")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "10"))

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Same normalisation as the validation transform in dataset.py, plus a resize so
# arbitrary uploads match the 32x32 inputs the model was trained on.
PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

app = FastAPI(title="CIFAR-10 Classifier")


def load_model():
    """Load the checkpoint written by train.py. Returns None if unavailable."""
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"checkpoint not found at {CHECKPOINT_PATH}", flush=True)
        return None

    try:
        net = get_model(architecture=MODEL_ARCH, num_classes=NUM_CLASSES)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        # Most often MODEL_ARCH disagrees with the architecture that was
        # trained. Stay up and report unhealthy rather than crash-looping.
        print(f"failed to load checkpoint: {exc}", flush=True)
        return None

    net.eval()
    print(f"loaded checkpoint from {CHECKPOINT_PATH}", flush=True)
    return net


model = load_model()


@app.get("/health")
def health():
    """200 once the model is loaded, 503 otherwise, so k8s probes can gate traffic."""
    global model
    if model is None:
        # Retry: the serving pod may start before the training Job has written
        # the checkpoint to the shared volume.
        model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "checkpoint": CHECKPOINT_PATH}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")

    tensor = PREPROCESS(pil_image).unsqueeze(0)

    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]

    top = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[top],
        "predicted_index": top,
        "confidence": round(float(probabilities[top]), 4),
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
