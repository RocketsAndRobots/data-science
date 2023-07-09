import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse


def segment_image(image, equiv_list):
    """
    Segment the image by checking which pixels belong to the same object/region.

    Parameters
    ----------
    img : numpy array
        Binary input image.
    equiv_list : python list
        List that stores the equivalencies of segmented regions.
    """
    label_count = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] != 0:
                above = (y - 1, x)
                left = (y, x - 1)
                left_up = (y - 1, x - 1)

                if image[left_up] != 0:
                    image[y, x] = image[left_up]
                elif image[left] != 0 and image[above] == 0:
                    image[y, x] = image[left]
                elif image[left] == 0 and image[above] != 0:
                    image[y, x] = image[above]
                elif image[left] != 0 and image[above] != 0:
                    if image[above] != image[left]:
                        label1, label2 = image[above], image[left]
                        equiv_list.update(
                            {label1: label2} if label1 > label2 else {label2: label1}
                        )
                    image[y, x] = image[above]
                else:
                    label_count += 1
                    image[y, x] = label_count
                    equiv_list.update({image[y, x]: image[y, x]})


def apply_equivalency_list(equiv_list, labeled_image):
    """
    Take the equivalency list to check which of the segmented regions belong together.

    Parameters
    ----------
    equiv_list : python list
            List that stores the equivalencies of segmented regions.
    labeled_image : numpy array
            Image that was already segmented but areas not merged yet
    """
    for y in range(labeled_image.shape[0]):
        for x in range(labeled_image.shape[1]):
            label = labeled_image[y, x]
            while True:
                new_label = equiv_list.get(label)
                if new_label is not None and new_label != label:
                    label = equiv_list.get(new_label)
                else:
                    break

            labeled_image[y, x] = label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sequential_labeling",
        description="An input image with homogeneous background will be segmented"
        + "and displayed.",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="images/example_3.png",
        type=str,
    )
    args = parser.parse_args()

    imported_image = cv2.imread(args.image_path)
    imported_gray_img = cv2.cvtColor(imported_image, cv2.COLOR_BGR2GRAY)
    thresh, imported_binary_img = cv2.threshold(
        imported_gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE
    )

    input_bin_img = np.array(imported_binary_img)
    equiv_list = dict()
    segment_image(input_bin_img, equiv_list)
    apply_equivalency_list(equiv_list, input_bin_img)

    print(f"Segmented {np.unique(input_bin_img).size - 1} components in the image")
    plt.figure("Input binary image")
    plt.imshow(imported_binary_img, cmap="binary")
    plt.figure("Segmented grayscale image")
    plt.imshow(input_bin_img, cmap="gist_yarg")
    plt.show()
