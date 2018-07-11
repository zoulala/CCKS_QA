import tensorflow as tf
from read_utils import TextConverter, batch_generator,load_origin_data,val_samples_generator
import os
import argparse  # 用于分析输入的超参数


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
    test_args.add_argument('--train_keep_prob', type=float, default=0.7,help='dropout rate during training')
    test_args.add_argument('--max_steps', type=int, default=100000,help='max steps to train')
    test_args.add_argument('--save_every_n', type=int, default=100,help='save the model every n steps')
    test_args.add_argument('--log_every_n', type=int, default=20,help='log to the screen every n steps')
    test_args.add_argument('--fc_activation', type=str, default='sigmoid', help='funciton of activated')
    test_args.add_argument('--feats', type=str, default='all', help='features of query')
    test_args.add_argument('--batch_norm', type=bool, default=False, help='standardization')
    test_args.add_argument('--op_method', type=str, default='adam', help='method of optimizer')
    test_args.add_argument('--checkpoint_path', type=str, default='models/thoth3/', help='checkpoint path')
    test_args.add_argument('--lr_decay', type=bool, default=False, help='standardization')




    return parser.parse_args(args)


## thoth 问答
args_in = '--file_name n26b200h400F ' \
          '--num_steps 26 ' \
          '--batch_size 200 ' \
          '--learning_rate 0.001 ' \
          '--hidden_size 400 ' \
          '--fc_activation sigmoid ' \
          '--op_method adam ' \
          '--max_steps 200000'.split()

FLAGS = parseArgs(args_in)



def main(_):
    model_path = os.path.join('models', FLAGS.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)





    if os.path.exists(os.path.join(model_path, 'converter.pkl')) or os.path.exists(os.path.join(model_path, 'QAs.pkl')) is False:
        print('词库文件不存在,创建...')
        QAs, text = load_origin_data('data/task3_train.txt')
        converter = TextConverter(text, 5000)
        converter.save_to_file(converter.vocab ,os.path.join(model_path, 'converter.pkl'))
        converter.save_to_file(QAs,os.path.join(model_path, 'QAs.pkl'))
    else:
        converter = TextConverter(filename=os.path.join(model_path, 'converter.pkl'))
        QAs = converter.load_obj(filename=os.path.join(model_path, 'QAs.pkl'))

    QA_arrs = converter.QAs_to_arrs(QAs, FLAGS.num_steps)


    thres = int(len(QA_arrs) * 0.9)
    train_samples = QA_arrs[:thres]
    val_samples = QA_arrs[thres:]

    train_g = batch_generator(train_samples, FLAGS.batch_size)
    val_g = val_samples_generator(val_samples)


    print('use embeding:',FLAGS.use_embedding)
    print('vocab size:',converter.vocab_size)

    from model3 import Model
    model = Model(converter.vocab_size,FLAGS,test=False, embeddings=None)

    # 继续上一次模型训练
    FLAGS.checkpoint_path = tf.train.latest_checkpoint(model_path)
    if FLAGS.checkpoint_path:
        model.load(FLAGS.checkpoint_path)

    model.train(train_g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                val_g
                )


if __name__ == '__main__':
    tf.app.run()