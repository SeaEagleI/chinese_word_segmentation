import Model,tools
import parameter as pm
import tensorflow as tf


def get_content2id(filename):
    _, content, __ = tools.get_words_content_label(filename)
    dictionary = tools.get_dictionary()
    content2id = tools.content2id(dictionary, content)
    return content2id


def get_word_list(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        text = fr.readlines()
    word_list = []
    for s in text:
        s = s.strip()
        word_list.append(s)
    return word_list


def predict(filename):
    content2id = get_content2id(filename)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/')
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)
    predict_label = []
    text_len = len(content2id)
    for i in range(0, text_len, pm.batch_size):
        predict_label.extend(model.predict(sess, content2id[i:min(i + pm.batch_size, text_len)]))
    return predict_label


def convert(sentence, corresponding_pre_label):
    convert_sentence = ''
    for i in range(len(corresponding_pre_label)):
        if corresponding_pre_label[i] == 2:
            # label == 'E'
            convert_sentence += sentence[i]
            convert_sentence += ' '
        elif corresponding_pre_label[i] == 3:
            # label == 'S'
            convert_sentence += ' '
            convert_sentence += sentence[i]
            convert_sentence += ' '
        else:
            convert_sentence += sentence[i]
    return convert_sentence


if __name__ == '__main__':
    model = Model.Model()
    predict_label = []
    # for i in range(1, 4):
    for i in range(1, 8):
        predict_label.extend(predict(pm.eva + str(i)))
    # predict_label.extend(predict('./data/pku_test.utf8'))
    sentences = []
    # for i in range(1, 4):
    for i in range(1, 8):
        sentences.extend(get_word_list(pm.eva + str(i)))
    # sentences.extend(get_word_list('./data/pku_test.utf8'))
    # with open(pm.eva, 'r', encoding='utf-8') as fr:
    #     text = fr.readlines()
    s = ''
    for i in range(len(sentences)):
        CWS = convert(sentences[i], predict_label[i])
        print(sentences[i])
        print(CWS)
        s += CWS
        s += '\n'

    with open('./data/msr_self.utf8', 'w', encoding='utf-8') as fw:
        fw.write(s)



