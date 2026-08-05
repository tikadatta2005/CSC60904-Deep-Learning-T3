from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from modules.architectures.Architecture import Architecture
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
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


# create model Architecture
model = Architecture()

def add_conv_layers(model, layers=1, skip_pool=0):
    # define in and out channels
    in_channels = 3
    out_channels = 8
    # size
    size = 112
    # skip pool increase by one for calculation
    skip_pool = skip_pool+1
    # save a trainable parameters
    total_conv_params = 0
    # loop each layers to add on model
    for layer in range(layers):
        # Convolutional Layer
        model.add(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # calculate trainable params
        total_conv_params += (((3*3*in_channels) + 1) * out_channels) + (2*out_channels)
        # Pooling Layer
        if  (layer + 1) % skip_pool==0:
            model.add(nn.MaxPool2d(2,2))
            size = size//2
        # update in and out channels
        if (layer<layers-1):
            in_channels = out_channels
            out_channels = out_channels*2
    # return total convolutional layer parameters
    return total_conv_params, out_channels, size

params, channels, size = add_conv_layers(model, 6, 1)

print(f"Total Convolutional Parameters: {params}")
print(f"Final Convolutional Out Channels: {channels}")
print(f"Final Convolutional Size: {size}")

flatten_size = channels * size * size

l1_param = flatten_size * channels + channels

l2_param = channels * 36 + 36

total_params = params + l1_param + l2_param

model.add(
    nn.Flatten(),
    nn.Linear(flatten_size, channels),
    nn.ReLU(),
    nn.Linear(channels, 36)
)

print(f"TOTAL TRAINABLE PARAMETERS = {total_params}")


checkpoint = torch.load(
    "./documentations/experiments/tika-6cnn-aug/models/epoch_29.pt",
    map_location=device
)


model.load_state_dict(
    checkpoint,
    strict=True
)


model = model.to(device)

model.eval()


transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])



classes = [
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व", "श",
    "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"
]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    image = transform(image)

    image = image.unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(image)

        probability = torch.softmax(
            output,
            dim=1
        ).item()
        
        confidence, prediction = torch.max(probability)
        
        if confidence <0.85: return {
            "prediction":None,
            "class_id":None
        }


    result = classes[prediction]


    print("Prediction:", result)


    return {
        "prediction": result,
        "class_id": pred
    }