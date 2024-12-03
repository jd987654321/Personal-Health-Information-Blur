import argparse
import json
import subprocess
import os
import cv2
import numpy as np
import shutil


model_results_path = os.getcwd() + '/model_results'
text_results_folder = os.getcwd() + '/text_results'
face_results_folder = os.getcwd() + '/face_results'
model_path = os.getcwd() + '/models'

retina_face_python = model_path + "/retina_face/re_face_venv/Scripts/python.exe"
retina_face_script = model_path + "/retina_face/script.py"
retina_face_results = face_results_folder + '/retina_face_detection_results.txt'

mtcnn_python = model_path + "/mtcnn/mtcnn/Scripts/python.exe"
mtcnn_script = model_path + "/mtcnn/script.py"
mtcnn_results = face_results_folder + '/mtcnn_detection_results.txt'

insightface_python = model_path + "/insightface/insightface/Scripts/python.exe"
insightface_script = model_path + "/insightface/script.py"
insightface_results = face_results_folder + '/insightface_detection_results.txt'

craft_python = model_path + "/craft/craft/Scripts/python.exe"
craft_script = model_path + "/craft/script.py"
craft_results = text_results_folder + '/craft_detection_results.txt'

easyocr_python = model_path + "/easyocr/easyocr/Scripts/python.exe"
easyocr_script = model_path + "/easyocr/script.py"
easyocr_result = text_results_folder + '/easyocr_results.txt'

paddleocr_python = model_path + "/paddle_ocr/paddle_venv/Scripts/python.exe"
paddleocr_script = model_path + "/paddle_ocr/script.py"
paddleocr_result = text_results_folder + '/paddleocr_results.txt'

def get_face_model_results(input_folder):
    subprocess.run([retina_face_python, retina_face_script, input_folder, retina_face_results])
    subprocess.run([mtcnn_python, mtcnn_script, input_folder, mtcnn_results])
    subprocess.run([insightface_python, insightface_script, input_folder, insightface_results])

def get_text_model_results(input_folder):
    subprocess.run([paddleocr_python, paddleocr_script, input_folder, paddleocr_result])
    subprocess.run([easyocr_python, easyocr_script, input_folder, easyocr_result])
    subprocess.run([craft_python, craft_script, input_folder, craft_results])

def move_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, image_file)
        if os.path.isfile(input_image_path):
            shutil.copy(input_image_path, output_folder)
            #print(f"Moved {image_file} to {output_folder}")
    print(f"\033[38;2;128;0;0mMoved images to {output_folder}\033[0m")



def blur_square(img, points, blur_kernel=(51, 51)):
    polygon_points = points
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 255)
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    result = img.copy()
    result[mask == 255] = blurred[mask == 255]

    return result


def draw_face_boxes(txt_files, output_folder):
    for txt_file in txt_files:
        #print(txt_file)
        with open(txt_file, "r") as f:
            lines = f.readlines()

        current_image = None
        bounding_boxes = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_image and bounding_boxes:
                    image_path = os.path.join(output_folder, current_image)
                    img = cv2.imread(image_path)

                    if img is None:
                        print(f"Error: Could not load image at {image_path}. Skipping file.")
                        continue

                    for bbox in bounding_boxes:
                        x1, y1, x2, y2 = bbox
                        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 0)
 

                    output_path = os.path.join(output_folder, current_image)
                    cv2.imwrite(output_path, img)
                    #print(f"Annotated and saved: {output_path}")

                current_image = None
                bounding_boxes = []
            elif current_image is None:
                current_image = line
            elif "no faces detected." in line.lower():
                #print(f"Skipping {current_image} due to no faces detected.")
                current_image = None
                bounding_boxes = []
            else:
                try:
                    bbox = tuple(map(int, line.split(',')))
                    bounding_boxes.append(bbox)
                except ValueError:
                    print(f"Skipping invalid bounding box data in {current_image}.")
                    bounding_boxes = []

        if current_image and bounding_boxes:
            image_path = os.path.join(output_folder, current_image)
            img = cv2.imread(image_path)

            if img is not None:
                for bbox in bounding_boxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                output_path = os.path.join(output_folder, current_image)
                cv2.imwrite(output_path, img)
                #print(f"Annotated and saved: {output_path}")
        print(f"\033[38;2;255;165;0mApplied results from {os.path.basename(txt_file)}\033[0m")


def draw_text_boxes(txt_files, output_folder):
    """
    Reads bounding boxes from multiple text files and applies them to images in the output folder.
    Processes each image only once, either when an empty line is encountered or at the end of the file.

    Args:
        txt_files (list): List of text file paths containing bounding box coordinates.
        output_folder (str): Path to the folder containing images and where annotated images are saved.

    Returns:
        None
    """
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()

        current_image = None
        bounding_boxes = []

        def process_image():
            nonlocal current_image, bounding_boxes
            if current_image and bounding_boxes:
                image_path = os.path.join(output_folder, current_image)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error: Could not load image at {image_path}. Skipping file.")
                    return

                for bbox in bounding_boxes:
                    points = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
                    img = blur_square(img, points)

                output_path = os.path.join(output_folder, current_image)
                cv2.imwrite(output_path, img)
                #print(f"Annotated and saved: {output_path}")

            current_image = None
            bounding_boxes = []

        for line in lines:
            line = line.strip()
            if not line:
                process_image()
            elif current_image is None:
                current_image = line
            elif "nothing detected" in line.lower():
                #print(f"Skipping {current_image} due to no text detected.")
                current_image = None
                bounding_boxes = []
            else:
                try:
                    bbox = tuple(map(int, line.split(',')))
                    if len(bbox) == 8:
                        bounding_boxes.append(bbox)
                    else:
                        print(f"Skipping invalid bounding box data in {current_image}: {line}")
                except ValueError:
                    print(f"Skipping invalid bounding box data in {current_image}: {line}")

        process_image()

        print(f"\033[38;2;255;165;0mApplied results from {os.path.basename(txt_file)}\033[0m")


if __name__ == "__main__":
    face_txt_files = [mtcnn_results, retina_face_results, insightface_results]
    text_txt_files = [paddleocr_result, easyocr_result, craft_results]
 
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder


    get_face_model_results(input_folder)
    get_text_model_results(input_folder)
    move_images(os.getcwd() + "/mall_images", os.getcwd() + "/results")
    draw_face_boxes(face_txt_files, output_folder)
    draw_text_boxes(text_txt_files, output_folder)
    

