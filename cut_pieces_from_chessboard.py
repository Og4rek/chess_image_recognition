import cv2
import os


def cut_roi(images):
    img_size = 784
    roi = []
    for img in images:
        for i in range(8):
            roi.append(img[0:98, 98*i:98+98*i])
            roi.append(img[img_size-98:img_size, 98*i:98+98*i])
        roi.append(img[98:196, 0:98])
        roi.append(img[588:686, 0:98])
        roi.append(img[98:196, 98:196])
        roi.append(img[588:686, 98:196])
    # roi = [   black_rook_1,   white_rook_1,
    #           black_knight_1, white_knight_1,
    #           black_bishop_1, white_bishop_1,
    #           black_quinn,    white_quinn,
    #           black_king,     white_king,
    #           black_bishop_2, white_bishop_2,
    #           black_knight_2, white_knight_2,
    #           black_rook_2,   white_rook_2,
    #           black_pawn_1,   white_pawn_1,
    #           black_pawn_1,   white_pawn_1, ...]
    return roi


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = img[1:784, 0:784]
            images.append(img)
    return images


if __name__ == '__main__':
    path_to_files = 'images/chess_board_images'
    images = load_images_from_folder(path_to_files)
    roi = cut_roi(images)
