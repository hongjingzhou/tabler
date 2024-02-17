from PIL import Image


def detect_tables(image_path):
    image = Image.open(image_path)
    image.convert("L")
    width, height = image.size
    print(f"Image size: {width}x{height}")


if __name__ == "__main__":
    detect_tables("./sample/tax.png")

