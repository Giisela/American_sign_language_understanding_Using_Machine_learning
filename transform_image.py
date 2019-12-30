import numpy as np
import cv2
import os
import csv
import traceback

from image_transformation import apply_image_transformation


def write_frame_to_file(frame, frame_label, writer):
    """
    Convert the multi-dimensonal array of the image to a one-dimensional one
    and write it to a file, along with its label.
    """
    print("Writing frame to file...")
    flattened_frame = frame.flatten()
    output_line = [frame_label] + np.array(flattened_frame).tolist()
    writer.writerow(output_line)
    print("Done!")


def init_transform_image():
    transformed_images_path = './dados/labels/images_transformed.csv'
    os.makedirs(os.path.dirname(transformed_images_path), exist_ok=True)
    with open(transformed_images_path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        training_images_labels_path = './dados/labels/training_images_labels.txt'
        with open(training_images_labels_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            print("\n\n" + line.strip())
            path, label = line.split()
            frame = cv2.imread(path)

            try:
                frame = apply_image_transformation(frame)
                write_frame_to_file(frame, label, writer)
            except Exception:
                exception_traceback = traceback.format_exc()
                print("Error applying image transformation to image ""'{}'".format(path))
                print(exception_traceback)
                continue
    cv2.destroyAllWindows()
    print("The program completed successfully !!")
