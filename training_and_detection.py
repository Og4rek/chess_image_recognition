import os
from git import Repo
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


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
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
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
    labels = [{'name': 'white_rook', 'id': 1}, {'name': 'white_knight', 'id': 2}, {'name': 'white_pawn', 'id': 3},
              {'name': 'white_bishop', 'id': 4}, {'name': 'white_king', 'id': 5}, {'name': 'white_queen', 'id': 6},
              {'name': 'black_rook', 'id': 7}, {'name': 'black_knight', 'id': 8}, {'name': 'black_pawn', 'id': 9},
              {'name': 'black_bishop', 'id': 10}, {'name': 'black_king', 'id': 11}, {'name': 'black_queen', 'id': 12}]

    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\''+label['name']+'\'\n')
            f.write('\tid:'+str(label['id'])+'\n')
            f.write('}\n')

    #create tfrecord
    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        git_url = 'https://github.com/nicknochnack/GenerateTFRecord'
        Repo.clone_from(git_url, paths['SCRIPTS_PATH'])

    command = [
        'python3.9 ' + files['TF_RECORD_SCRIPT'] + ' -x ' + os.path.join(paths['IMAGE_PATH'], 'train') + ' -l ' + files['LABELMAP'] + ' -o ' + os.path.join(paths['ANNOTATION_PATH'], 'train.record'),
        'python3.9 ' + files['TF_RECORD_SCRIPT'] + ' -x ' + os.path.join(paths['IMAGE_PATH'], 'test') + ' -l ' + files['LABELMAP'] + ' -o ' + os.path.join(paths['ANNOTATION_PATH'], 'test.record'),
        ]

    if not os.path.exists(os.path.join(paths['ANNOTATION_PATH'], 'test.record')):
        for cmd in command:
            os.system(cmd)

    # Copy Model Config to Training Folder
    command = 'cp ' + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + ' ' + os.path.join(paths['CHECKPOINT_PATH'])
    os.system(command)

    # Update config
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 1
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

    # Training the model
    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python3.9 " + TRAINING_SCRIPT + " --model_dir=" + paths['CHECKPOINT_PATH'] + " --pipeline_config_path=" + files['PIPELINE_CONFIG'] + " --num_train_steps=2000"
    print(command)
    os.system(command)

    # Evaluation
    command = "python3.9 {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
    print(command)
    os.system(command)

