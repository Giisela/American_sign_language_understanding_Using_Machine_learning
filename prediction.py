import cv2
import traceback
from sklearn.externals import joblib
from image_transformation import apply_image_transformation


def init_prediction(model_name):

    print("Using model {}...".format(model_name))

    model_serialized_path = './dados/labels/result/model-serialized-{}.pkl'.format(model_name)

    testing_images_labels_path = './dados/labels/testing_images_labels.txt'
    with open(testing_images_labels_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line:
                continue
            image_path, image_label = line.split()
            frame = cv2.imread(image_path)
            try:
                frame = apply_image_transformation(frame)
                frame_flattened = frame.flatten()
                classifier_model = joblib.load(model_serialized_path)
                predicted_labels = classifier_model.predict([frame_flattened])
                predicted_label = predicted_labels[0]
                print('"{}" {} ---> {}'.format(image_path, image_label, predicted_label))
                if image_label != predicted_label:
                    log_msg = "Incorrect prediction '{}' instead of '{}'\n)"
                    print(log_msg.format(predicted_label, image_label))
                    cv2.waitKey(5000)
            except Exception:
                exception_traceback = traceback.format_exc()
                print("Error applying image transformation to image "
                             "'{}'".format(image_path))
                print(exception_traceback)
                continue
    cv2.destroyAllWindows()
    print("The program completed successfully !!")
