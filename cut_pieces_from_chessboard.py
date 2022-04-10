import cv2
import os


def roi_save(roi):
    path_to_roi_save = 'images/pieces'
    pieces_dir_name = ['black_rook',    'white_rook',
                       'black_knight',  'white_knight',
                       'black_bishop',  'white_bishop',
                       'black_quinn',   'white_quinn',
                       'black_king',    'white_king',
                       'black_pawn',    'white_pawn']
    if not os.path.exists(path_to_roi_save):
        os.makedirs(path_to_roi_save)
    for dir in pieces_dir_name:
        if not os.path.exists(path_to_roi_save+'/'+dir):
            os.makedirs(path_to_roi_save+'/'+dir)
    for i in range(int(len(roi)/20)):
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[0]+'/'+str(2*i)+'.png',    roi[0+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[1]+'/'+str(2*i)+'.png',    roi[1+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[2]+'/'+str(2*i)+'.png',    roi[2+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[3]+'/'+str(2*i)+'.png',    roi[3+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[4]+'/'+str(2*i)+'.png',    roi[4+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[5]+'/'+str(2*i)+'.png',    roi[5+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[6]+'/'+str(i)+'.png',      roi[6+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[7]+'/'+str(i)+'.png',      roi[7+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[8]+'/'+str(i)+'.png',      roi[8+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[9]+'/'+str(i)+'.png',      roi[9+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[4]+'/'+str(2*i+1)+'.png',  roi[10+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[5]+'/'+str(2*i+1)+'.png',  roi[11+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[2]+'/'+str(2*i+1)+'.png',  roi[12+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[3]+'/'+str(2*i+1)+'.png',  roi[13+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[0]+'/'+str(2*i+1)+'.png',  roi[14+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[1]+'/'+str(2*i+1)+'.png',  roi[15+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[10]+'/'+str(2*i)+'.png',   roi[16+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[11]+'/'+str(2*i)+'.png',   roi[17+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[10]+'/'+str(2*i+1)+'.png', roi[18+i*20])
        cv2.imwrite(path_to_roi_save+'/'+pieces_dir_name[11]+'/'+str(2*i+1)+'.png', roi[19+i*20])



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
    roi_save(roi)