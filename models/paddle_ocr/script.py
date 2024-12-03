from paddleocr import PaddleOCR
import os
import argparse

def main(input_folder, output_txt_file):
    """
    Runs text detection on images using PaddleOCR and saves results to a text file.

    Args:
        input_folder (str): Path to the folder containing images.
        output_txt_file (str): Path to the output text file.

    Returns:
        None
    """
    # Initialize PaddleOCR for detection
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Customize language if needed
    print(output_txt_file)
    # Open the output text file for writing
    with open(output_txt_file, "w") as f:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
                image_path = os.path.join(input_folder, filename)
                try:
                    # Run OCR on the image
                    result = ocr.ocr(image_path, det=True, rec=False)  # Detection only

                    # Write the filename to the output file
                    f.write(f"{filename}\n")
                    #print("checkpoint 1")
                    # Write detected text bounding boxes
                    if(len(result[0]) != 0):
                        for line in result[0]:
                            #bbox = line[0]  # Bounding box coordinates
                            #print(line)
                            #print("checkpoint 2")
                            #print(bbox)
                            coords = ", ".join([f"{int(bbox[0])}, {int(bbox[1])}" for bbox in line])
                            #print("checkpoint 3")
                            f.write(f"{coords}\n")
                    else:
                        f.write("Nothing detected\n")
                    
                    # Add a blank line after each image's results
                    f.write("\n")
                    #print(f"Processed {filename} and saved results.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    print("\033[32mProcessed and saved results for PaddleOCR\033[0m")

# Example usage
#input_folder = "images"  # Folder containing input images
#output_txt_file = "text_detection_results.txt"  # Output text file for saving results

#save_text_detection_results(input_folder, output_txt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_txt_file", type=str)
    
    args = parser.parse_args()
    main(input_folder=args.input_folder, output_txt_file=args.output_txt_file)