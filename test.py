# ref:https://github.com/hzy46/Char-RNN-TensorFlow
import tensorflow as tf
from read_utils import TextConverter,load_origin_data,val_samples_generator,test_samples_generator
import os

# FLAGS = tf.flags.FLAGS
#
# tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
# tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
# tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
# tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
# tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
# tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
# tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
# tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')
#




import argparse # 用于分析输入的超参数

def parseArgs(args):
    """
    Parse 超参数
    Args:
        args (list<stir>): List of arguments.
    """


    parser = argparse.ArgumentParser()
    test_args = parser.add_argument_group('test超参数')
    test_args.add_argument('--file_name', type=str, default='default',help='name of the model')
    test_args.add_argument('--batch_size', type=int, default=100,help='number of seqs in one batch')
    test_args.add_argument('--num_steps', type=int, default=100,help='length of one seq')
    test_args.add_argument('--hidden_size', type=int, default=128,help='size of hidden state of lstm')
    test_args.add_argument('--num_layers', type=int, default=2,help='number of lstm layers')
    test_args.add_argument('--use_embedding', type=bool, default=False,help='whether to use embedding')
    test_args.add_argument('--embedding_size', type=int, default=128,help='size of embedding')
    test_args.add_argument('--learning_rate', type=float, default=0.001,help='learning_rate')
    test_args.add_argument('--train_keep_prob', type=float, default=1,help='dropout rate during training')
    test_args.add_argument('--max_steps', type=int, default=100000,help='max steps to train')
    test_args.add_argument('--save_every_n', type=int, default=1000,help='save the model every n steps')
    test_args.add_argument('--log_every_n', type=int, default=10,help='log to the screen every n steps')
    test_args.add_argument('--fc_activation', type=str, default='sigmoid', help='funciton of activated')
    test_args.add_argument('--feats', type=str, default='all', help='features of query')
    test_args.add_argument('--batch_norm', type=bool, default=False, help='standardization')
    test_args.add_argument('--op_method', type=str, default='adam', help='method of optimizer')
    test_args.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path')
    test_args.add_argument('--lr_decay', type=bool, default=False, help='standardization')

    return parser.parse_args(args)

## thoth
args_in = '--file_name n26b200h400 ' \
          '--num_steps 26 ' \
          '--batch_size 200 ' \
          '--hidden_size 400 ' \
          '--fc_activation sigmoid '.split()


FLAGS = parseArgs(args_in)


def main(_):


    model_path = os.path.join('models', FLAGS.file_name)
    if os.path.isdir(model_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(model_path)

    converter = TextConverter(filename=os.path.join(model_path, 'converter.pkl'))
    # QAs = converter.load_obj(filename=os.path.join(model_path, 'QAs.pkl'))
    QAs, text = load_origin_data('data/task3_dev.txt')
    testQAs_to_arrs = converter.testQAs_to_arrs(QAs, FLAGS.num_steps)

    test_samples = testQAs_to_arrs

    print('use embeding:',FLAGS.use_embedding)
    print('vocab size:',converter.vocab_size)

    from model3 import Model

    with open(model_path+'/submission.csv', 'w') as file:
        file.write('test_id,result' + '\n')
    batchsize = 1000
    for i in range(0,len(test_samples), batchsize):  # 内存不足 分批test
        print('>>>>:',i,'/',len(test_samples))
        test_g = test_samples_generator(test_samples[i:i+batchsize])

        model = Model(converter.vocab_size, FLAGS, test=False, embeddings=None)

        model.load(FLAGS.checkpoint_path)

        model.test(test_g, model_path)
    print('finished!')

if __name__ == '__main__':
    tf.app.run()
