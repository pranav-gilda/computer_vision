import torch
import torch.nn as nn
from torchvision import models

# --- PASTE YOUR FCN CLASS DEFINITION HERE ---
class FCN(nn.Module):
    def __init__(self, num_classes=2):
        super(FCN, self).__init__()
        # Note: Using pretrained=False as weights are loaded from file
        vgg = models.vgg16() # Removed weights parameter for compatibility
        self.encoder = vgg.features
        
        # The decoder part is exactly the same
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, num_classes, 3, 2, 1, 1))
            
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Main Conversion Logic ---
def convert_model():
    # --- Configuration ---
    PYTORCH_MODEL_PATH = "fcn_road_segmentation_v2.pth"
    ONNX_MODEL_PATH = "fcn_road_segmentation_v2.onnx"
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 576

    print(f"Loading PyTorch model from: {PYTORCH_MODEL_PATH}")

    # Initialize the model architecture
    model = FCN(num_classes=2)
    
    # Load the trained weights. Use map_location for CPU compatibility.
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()
    print("Model loaded successfully.")

    # Create a dummy input tensor with the correct dimensions
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, requires_grad=True)

    print(f"Exporting model to ONNX format at: {ONNX_MODEL_PATH}")
    
    # Export the model
    torch.onnx.export(model,                           # model being run
                      dummy_input,                     # model input
                      ONNX_MODEL_PATH,                 # where to save the model
                      export_params=True,              # store the trained weights
                      opset_version=11,                # the ONNX version
                      do_constant_folding=True,        # execute constant folding for optimization
                      input_names = ['input'],         # the model's input names
                      output_names = ['output'])       # the model's output names

    print("\n--- Conversion Successful! ---")
    print(f"Your model has been saved as '{ONNX_MODEL_PATH}'")

if __name__ == '__main__':
    convert_model()