import os
from mtcnn import MTCNN
import cv2
import argparse

def main(input_dir, output_txt_file):
    # Paths to input folder and output text file
    input_folder = input_dir
    #output_dir = 'C:/Users/jacob/Downloads/glendor_final'
    #output_txt_file = os.path.join(output_dir, "mtcnn_detection_results.txt")
    #output_txt_file = "bounding_boxes.txt"

    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Open the output text file for writing
    with open(output_txt_file, "w") as f:
        # Process each image in the input folder
        for filename in os.listdir(input_folder):
            # Only process image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load and process the image
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue  # Skip non-image files
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = detector.detect_faces(image_rgb)
                
                # Write the filename to the output file
                f.write(f"{filename}\n")
                
                # Write bounding box coordinates
                print(type(results))
                if len(results) != 0:
                    for face in results:
                        x, y, width, height = face['box']
                        f.write(f"{x}, {y}, {x+width}, {y+height}\n")
                else:
                    f.write("No faces detected.\n")

                # Add a blank line after each image's results
                f.write("\n")

    #print(f"Bounding box coordinates saved to {output_txt_file}.")
    print("\033[32mProcessed and saved results for Insight Face\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_txt_file", type=str)
    
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_txt_file=args.output_txt_file)
