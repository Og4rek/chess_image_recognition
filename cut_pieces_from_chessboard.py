import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    path_to_files = 'images/chess_board_images'
    images = load_images_from_folder(path_to_files)
