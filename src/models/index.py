
from baseModels.resnet_regression import ResNetModel
from src.XAI.VanillaSaliency import VanillaSaliency

model = ResNetModel(data_dir='../artificial_data/noisy_generated_images', num_epochs=1)
model.load_data()
model.train()
model.evaluate()

# Initialize XAI instance
xai_resnet = VanillaSaliency(modelWrapper=model.model, device=model.device)

# Fetch an image and its label from the test data
input_image, input_label = model.get_single_test_image(index=0)  # You can change index to get different images

# Check if data was retrieved
if input_image is not None:
    # Generate and view the saliency map for the selected image and label
    xai_resnet.generate_map(input_image, input_label)
