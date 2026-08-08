from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from modules.architectures.Architecture import Architecture

from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = Architecture()


def add_conv_layers(model, layers=1, skip_pool=0):

    # Initial channels
    in_channels = 3
    out_channels = 8

    # Input image size
    size = 112

    # Used for pooling calculation
    skip_pool = skip_pool + 1

    # Parameter counter
    total_conv_params = 0

    for layer in range(layers):

        model.add(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        total_conv_params += (
            ((3 * 3 * in_channels) + 1) * out_channels
            + (2 * out_channels)
        )

        if (layer + 1) % skip_pool == 0:

            model.add(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

            size = size // 2

        if layer < layers - 1:

            in_channels = out_channels
            out_channels = out_channels * 2

    return (
        total_conv_params,
        out_channels,
        size
    )

params, channels, size = add_conv_layers(
    model,
    layers=6,
    skip_pool=1
)

print(
    f"Total Convolutional Parameters: {params}"
)

print(
    f"Final Convolutional Out Channels: {channels}"
)

print(
    f"Final Convolutional Size: {size} x {size}"
)



flatten_size = channels * size * size

l1_param = (
    flatten_size * channels
    + channels
)

l2_param = (
    channels * 36
    + 36
)

total_params = (
    params
    + l1_param
    + l2_param
)


model.add(
    nn.Flatten(),

    nn.Linear(
        flatten_size,
        channels
    ),

    nn.ReLU(),

    nn.Linear(
        channels,
        36
    )
)


print(
    f"TOTAL TRAINABLE PARAMETERS = {total_params}"
)


checkpoint_path = (
    "./documentations/experiments/"
    "tika-6cnn-aug/models/epoch_29.pt"
)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)


model.load_state_dict(
    checkpoint,
    strict=True
)


# Move model to GPU/CPU
model = model.to(device)

# Evaluation mode
model.eval()


print("Model loaded successfully.")


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])


classes = [
 'क',
 'क्ष',
 'ख',
 'ग',
 'घ',
 'ङ',
 'च',
 'छ',
 'ज',
 'ज्ञ',
 'झ',
 'ञ',
 'ट',
 'ठ',
 'ड',
 'ढ',
 'ण',
 'त',
 'त्र',
 'थ',
 'द',
 'ध',
 'न',
 'प',
 'फ',
 'ब',
 'भ',
 'म',
 'य',
 'र',
 'ल',
 'व',
 'श',
 'ष',
 'स',
 'ह'
 ]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Save original dimensions for debugging
    original_width, original_height = image.size

    image_tensor = transform(image)
    print(
        f"Original image: "
        f"{original_width} x {original_height}"
    )

    print(
        f"Tensor before unsqueeze: "
        f"{image_tensor.shape}"
    )

    image_tensor = image_tensor.unsqueeze(0)

    print(
        f"Tensor after unsqueeze: "
        f"{image_tensor.shape}"
    )

    # Move tensor to same device as model
    image_tensor = image_tensor.to(device)


    with torch.no_grad():

        output = model(image_tensor)

        # Convert logits to probabilities
        probability = torch.softmax(
            output,
            dim=1
        )

        # Get highest probability
        confidence, prediction = torch.max(
            probability,
            dim=1
        )


    confidence = confidence.item()
    prediction = prediction.item()

    result = classes[prediction]

    print(
        f"Prediction: {result}"
    )

    print(
        f"Class ID: {prediction}"
    )

    print(
        f"Confidence: {confidence:.6f}"
    )

    if confidence < 0.80:

        return {
            "prediction": None,
            "class_id": None,
            "confidence": confidence
        }

    return {
        "prediction": result,
        "class_id": prediction,
        "confidence": confidence
    }