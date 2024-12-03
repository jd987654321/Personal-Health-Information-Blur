import os
import argparse
from retinaface import RetinaFace


def main(input_folder, output_txt_file):
    # Define the input and output directories
    input_dir = input_folder
    #output_dir = 'C:/Users/jacob/Downloads/glendor_final'
    #output_file = os.path.join(output_dir, "retina_face_detection_results.txt")
    output_file = output_txt_file

    # Ensure the output directory exists
    #os.makedirs(output_dir, exist_ok=True)

    # Open the output file for writing
    with open(output_file, 'w') as file:
        # Process each image in the input directory
        #file.write("retina face")
        for filename in os.listdir(input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_dir, filename)

                # Perform face detection
                faces = RetinaFace.detect_faces(image_path)

                # Write the image name to the file
                file.write(f"{filename}\n")

                # Check if any faces were detected
                if isinstance(faces, dict):
                    for face_id, face_info in faces.items():
                        # Get bounding box coordinates
                        facial_area = face_info['facial_area']
                        x1, y1, x2, y2 = facial_area

                        # Write the bounding box to the file
                        file.write(f"{x1}, {y1}, {x2}, {y2}\n")
                else:
                    file.write("No faces detected.\n")

                # Add a blank line between entries for different images
                file.write("\n")

    #print(f"Detection results saved to {output_file}")
    print("\033[32mProcessed and saved results for Retina Face\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_txt_file", type=str)
    
    args = parser.parse_args()
    main(input_folder=args.input_folder, output_txt_file=args.output_txt_file)
