import os
import torch
import cv2
import numpy as np
from craft import CRAFT
import craft_utils
import imgproc
from torch.autograd import Variable
import argparse

# Load the CRAFT model
def load_craft_model(weights_path, use_cuda):
    from collections import OrderedDict

    print("Loading CRAFT model...")
    net = CRAFT()

    # Load weights
    state_dict = torch.load(weights_path, map_location='cuda' if use_cuda else 'cpu')

    # Remove "module." prefix if necessary
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v

    # Load updated state dictionary into the model
    net.load_state_dict(new_state_dict)

    # Move to GPU if needed
    if use_cuda:
        net = net.cuda()

    net.eval()
    return net


# Process a single image and detect text regions
# Process a single image and detect text regions
def process_image(net, image_path, canvas_size=1280, mag_ratio=1.5, text_threshold=0.7, link_threshold=0.4, low_text=0.4, use_cuda=False):
    # Load and preprocess the image
    image = imgproc.loadImage(image_path)
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, canvas_size, mag_ratio=mag_ratio, interpolation=cv2.INTER_LINEAR)
    ratio_h, ratio_w = 1 / target_ratio, 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] -> [c, h, w]
    x = Variable(x.unsqueeze(0))  # Add batch dimension
    if use_cuda:
        x = x.cuda()

    # Perform inference
    with torch.no_grad():
        y, _ = net(x)

    # Extract score maps
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # Post-process to get bounding boxes
    boxes, _ = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes


# Main script
def main(input_folder, output_txt_file):
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = script_dir + "\craft_mlt_25k.pth"  # Path to the CRAFT model weights
    #input_folder = "mall_images"  # Folder containing input images
    #output_txt_file = "resultsresultsresults.txt"  # Single output file for all results

    # Check for CUDA
    use_cuda = torch.cuda.is_available()

    # Load the model
    net = load_craft_model(model_path, use_cuda)

    # Open the output text file
    with open(output_txt_file, "w") as f:
        # Process all images in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_folder, filename)

                # Detect bounding boxes
                boxes = process_image(net, input_path, use_cuda=use_cuda)

                # Write results to the text file
                f.write(f"{filename}\n")
                if len(boxes) != 0:
                    for box in boxes:
                        coords = ", ".join([f"{int(p[0])}, {int(p[1])}" for p in box])
                        f.write(f"{coords}\n")
                else:
                    f.write("Nothing detected.\n")
                f.write("\n")  # Blank line between results for each image

                #print(f"Processed: {filename}")
    print("\033[32mProcessed and saved results for CRAFT\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_txt_file", type=str)
    
    args = parser.parse_args()
    main(input_folder=args.input_folder, output_txt_file=args.output_txt_file)
