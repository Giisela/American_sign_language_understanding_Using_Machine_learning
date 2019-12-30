from img_labels import init_img_labels
from transform_image import init_transform_image
from train_svm_classifier import init_train_svm_classifier
from prediction import init_prediction


def main():
    print("Start img labels train...")
    init_img_labels("train")
    print("End img labels train...")

    print("Start transform_image...")
    init_transform_image()
    print("End transform_image...")

    print("Start train svm classifier...")
    init_train_svm_classifier()
    print("End train svm classifier...")

    print("Start img labels test...")
    init_img_labels("test")
    print("End img labels test...")

    print("Start prediction...")
    init_prediction("svm")
    print("End prediction...")

    print("The program completed successfully !!")


if __name__ == '__main__':
    main()
