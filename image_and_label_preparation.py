import os
from git import Repo

if __name__ == '__main__':

    labels = ['white_rook', 'white_knight', 'white_pawn', 'white_bishop', 'white_king', 'white_queen',
              'black_rook', 'black_knight', 'black_pawn', 'black_bishop', 'black_king', 'black_queen']
    number_img = 6

    images_path = os.path.join('tensorflow', 'workspace', 'images', 'colletedimages')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    for label in labels:
        path = os.path.join(images_path, label)
        if not os.path.exists(path):
            os.mkdir(path)

    label_img_path = os.path.join('tensorflow', 'label_img')
    if not os.path.exists(label_img_path):
        os.makedirs(label_img_path)
        git_url = 'https://github.com/tzutalin/labelImg.git'
        Repo.clone_from(git_url, label_img_path)

    os.chdir(label_img_path)
    os.system('ls')
    os.system('make qt5py3')
    os.system('python3.9 labelImg.py')