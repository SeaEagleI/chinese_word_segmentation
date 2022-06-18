import os

import tools
import parameter as pm
import tensorflow as tf
import Model


def init_dict(filename):
    words_set, _, __ = tools.get_words_content_label(filename)
    tools.decode_word(words_set)


def train(model):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    dictionary = tools.get_dictionary()

    _, content, label = tools.get_words_content_label(pm.train)
    content_id_train = tools.content2id(dictionary, content)
    label_id_train = tools.label2id(label)
    _, content, label = tools.get_words_content_label(pm.test)
    content_id_test = tools.content2id(dictionary, content)
    label_id_test = tools.label2id(label)

    for epoch in range(pm.epochs):
        print('第 ', epoch+1, '次')
        num_batch = int(len(content_id_train) / pm.batch_size) + 1
        for x_batch, y_batch in  tools.next_batch(content_id_train, label_id_train):
            # next_batch 是一个产生器 generator
            x_batch_with_pad, origin_x_batch_len = tools.sentence_process(x_batch)
            y_batch_with_pad, origin_y_batch_len = tools.sentence_process(y_batch)
            feed_dict = model.feed_data(x_batch_with_pad, y_batch_with_pad, origin_x_batch_len, pm.keep_pro)
            _, global_step, loss = session.run([model.optimizer, model.global_step, model.loss], feed_dict=feed_dict)

            if global_step % 100 == 0:
                test_loss = model.test(session, content_id_test, label_id_test)
                print('global_step: ', global_step, 'train_loss:', loss, 'test_loss', test_loss)

            if global_step % (2 * num_batch) == 0:
                print('Saving Model...')
                saver.save(session, save_path='./checkpoints/', global_step=global_step)
            pm.learning_rate *= pm.lr


if __name__ == '__main__':
    if not os.path.exists('./data/dictionary.pkl'):
        init_dict(pm.train)
    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints/')

    model = Model.Model()
    train(model)
