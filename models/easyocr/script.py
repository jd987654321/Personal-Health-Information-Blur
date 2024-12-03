import os
import easyocr
import argparse

def save_easyocr_results_to_txt(input_folder, output_txt_file, languages=['en']):
    """
    Processes images in a folder using EasyOCR, saves bounding box coordinates
    and detected text to a text file in a structured format.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_txt_file (str): Path to the output text file.
        languages (list): List of languages for OCR (default is English ['en']).
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(languages)

    # Open the output text file for writing
    with open(output_txt_file, "w") as f:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
                input_path = os.path.join(input_folder, filename)

                try:
                    # Perform OCR
                    results = reader.readtext(input_path)

                    # Write the filename to the text file
                    f.write(f"{filename}\n")

                    # Write the bounding box coordinates
                    if len(results) != 0:
                        for (bbox, text, confidence) in results:
                            coords = ", ".join([f"{int(point[0])}, {int(point[1])}" for point in bbox])
                            f.write(f"{coords}\n")
                    else: 
                        f.write("Nothing detected\n")

                    # Add a blank line after each image's results
                    f.write("\n")
                    #print(f"Processed {filename} and saved results.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    print("\033[32mProcessed and saved results for easyOCR\033[0m")


# Example usage
#input_folder = "images"  # Input folder containing images
#output_txt_file = "easyocr_results.txt"  # Output text file for saving results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_txt_file", type=str)
    
    args = parser.parse_args()
    save_easyocr_results_to_txt(input_folder=args.input_folder, output_txt_file=args.output_txt_file)
