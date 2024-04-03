
from models.baseModels.resnet_regression import ResNetModel
from src.XAI.VanillaSaliency import VanillaSaliency

resnet = ResNetModel(data_dir='../datasets/artificial_data/noisy_generated_images', num_epochs=1)
resnet.load_data()
resnet.train()
resnet.evaluate()

# Initialize XAI instance
xai_resnet = VanillaSaliency(modelWrapper=resnet, device=resnet.device)

# Fetch an image and its label from the test data
input_image, input_label = resnet.get_single_test_image(index=0)  # You can change index to get different images

# Check if data was retrieved
if input_image is not None:
    # Generate and view the saliency map for the selected image and label
    xai_resnet.generate_saliency_map(input_image, input_label)
