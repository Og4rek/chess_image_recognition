import os
from git import Repo
import object_detection

if __name__ == '__main__':
    # names of models
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('tensorflow', 'scripts'),
        'APIMODEL_PATH': os.path.join('tensorflow', 'models'),
        'ANNOTATION_PATH': os.path.join('tensorflow', 'workspace', 'annotations'),
        'IMAGE_PATH': os.path.join('tensorflow', 'workspace', 'images'),
        'MODEL_PATH': os.path.join('tensorflow', 'workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('tensorflow', 'workspace', 'pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('tensorflow', 'protoc')
    }

    files = {
        'PIPELINE_CONFIG': os.path.join('tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABALMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    # make dirs for data
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    # clone api model repo
    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        git_url = 'https://github.com/tensorflow/models'
        Repo.clone_from(git_url, paths['APIMODEL_PATH'])

    #os.system('brew list protobuf || brew install protobuf')
    #os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python3.9 -m pip install .')

    #VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    #command = 'python3.9 ' + VERIFICATION_SCRIPT
    #os.system(command)


    # download pretrained model
    command = ['wget ' + PRETRAINED_MODEL_URL,
               'mv ' + PRETRAINED_MODEL_NAME + '.tar.gz ' + paths['PRETRAINED_MODEL_PATH'],
               'cd ' + paths['PRETRAINED_MODEL_PATH'] + '&& tar -zxvf ' + PRETRAINED_MODEL_NAME + '.tar.gz']

    #os.system('brew list wget || brew install wget')
    if not os.path.exists(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME)):
        for cmd in command:
            os.system(cmd)

    # create label map

