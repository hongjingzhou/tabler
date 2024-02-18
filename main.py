import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection


def plot_image(image):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def plot_results(image, model, scores, labels, boxes):
    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    for score, label, box, color in zip(scores.tolist(), labels.tolist(), boxes.tolist(), COLORS * 100):
        x_min, y_min, x_max, y_max = box
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color=color, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(x_min, y_min, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()


def detect_tables(image):
    feature_extractor = DetrFeatureExtractor()
    encoding = feature_extractor(images=image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    with torch.no_grad():
        outputs = model(**encoding)

    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.7, target_sizes=[(image.size[1], image.size[0])])[0]

    plot_results(image, model, results["scores"], results["labels"], results["boxes"])


def recognize_tables(image):
    feature_extractor = DetrFeatureExtractor()
    encoding = feature_extractor(images=image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.7, target_sizes=target_sizes)[0]

    plot_results(image, model, results["scores"], results["labels"], results["boxes"])


def main():
    filepaths = ["./sample/tax.png", "./sample/pub.png", "./sample/tsr.png"]
    for filepath in filepaths:
        image = Image.open(filepath).convert("RGB")
        plot_image(image)
        detect_tables(image)
        recognize_tables(image)


if __name__ == "__main__":
    matplotlib.use('qtagg')
    main()
