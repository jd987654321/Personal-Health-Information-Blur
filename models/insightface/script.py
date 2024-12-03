import insightface
import cv2
import os
import argparse

def main(input_folder, output_txt_file):
# Load the InsightFace model
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0)  # Adjust `ctx_id` as needed (e.g., -1 for CPU)

    # Specify the input folder and output text file
    #input_folder = "images"
    #output_dir = 'C:/Users/jacob/Downloads/glendor_final'
    #output_txt_file = os.path.join(output_dir, "insightface_detection_results.txt")

    # Open the text file for writing
    with open(output_txt_file, "w") as f:
        # Process each image in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
                image_path = os.path.join(input_folder, filename)
                img = cv2.imread(image_path)

                # Check if the image was loaded correctly
                if img is None:
                    print(f"Error: Could not load image at {image_path}. Skipping file.")
                    continue

                # Detect and analyze faces in the image
                faces = model.get(img)
                
                # Write the filename to the output file
                f.write(f"{filename}\n")

                # Write bounding box coordinates
                if len(faces) != 0:
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        f.write(f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n")
                else:
                    f.write("No faces detected.\n")
                
                # Add a blank line after each image's results
                f.write("\n")

   # print(f"Bounding box coordinates saved to {output_txt_file}.")
    print("\033[32mProcessed and saved results for Insight Face\033[0m")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_txt_file", type=str)
    
    args = parser.parse_args()
    main(input_folder=args.input_folder, output_txt_file=args.output_txt_file)