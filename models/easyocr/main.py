import os
import easyocr
import cv2

def annotate_images_with_easyocr(input_folder, output_folder, languages=['en']):
    """
    Processes images in a folder using EasyOCR, overlays results on the images,
    and saves the annotated images to a different folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save annotated images.
        languages (list): List of languages for OCR (default is English ['en']).
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(languages)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error loading image {filename}. Skipping.")
                continue

            # Perform OCR
            results = reader.readtext(input_path)

            # Annotate the image with OCR results
            for (bbox, text, confidence) in results:
                # Get bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                # Draw the bounding box
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

                # Put the recognized text above the bounding box
                cv2.putText(image, f"{text} ({confidence:.2f})", 
                            (top_left[0], top_left[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Save the annotated image to the output folder
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved: {output_path}")

# Example usage
input_folder = "images"  # Input folder containing images
output_folder = "results"  # Output folder for annotated images

annotate_images_with_easyocr(input_folder, output_folder)
