# -*- coding: utf-8 -*-
"""
本文采用组合 BiLSTM+CNN的方法输出特征，组合(concat)后，分类
BiLSTM和CNN是并行的级别
"""
import tensorflow as tf
import parameter as pm
import tools as t
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
import keras as kr

class Model:
    def __init__(self):
        self.x_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_input')
        self.y_intput = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_input')
        self.seq_length = tf.placeholder(dtype=tf.int32, name='seq_length')
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_probability')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.biLSTMCRF()

    def biLSTMCRF(self):
        with tf.name_scope('embedding'):
            embedding = tf.Variable(tf.truncated_normal([pm.vacab_size, pm.embedding_size], -0.25, 0.25))
            embedding_input = tf.nn.embedding_lookup(embedding, self.x_input)
            self.embedding = tf.nn.dropout(embedding_input, keep_prob=self.keep_pro)
        with tf.name_scope('BiLSTM'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(pm.hidden_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(pm.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, self.keep_pro)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, self.keep_pro)

            # outputs = (output_fw, output_bw)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.embedding
                                                         , dtype=tf.float32)
            # output_fw 由 (batch_size, max_seq_len, hidden_size)
            # 含义：批次中的每个句子的每个词在每个LSTMCELL中输出一个结果，
            # 每个句子组成一个(max_seq_len,hidden_size)的矩阵

            # 连接 output_fw 和 output_bw 每批次中所有时刻输出的fw和bw中的output
            outputs = tf.concat(outputs, 2)
            shape = tf.shape(outputs)
            # 原来的形状为(batch_size, max_seq_len, 2*hidden_size)
            #            (64, 600, 2*128)
            # 转换后变为 (batch_size, 2*pm.hidden_size)
            # 批次中bilstm所有时刻的output
            output = tf.reshape(outputs, [-1, 2 * pm.hidden_size])

        with tf.name_scope('CNN'):
            # 加入cnn 层
            # 使用CNN导致形变，提前padding 补0，注意keras中1维度padding补的是句子长度。不会补充词向量的长度
            self.embedding = kr.layers.ZeroPadding1D(padding=pm.half_kernel_size)(self.embedding)
            conv = tf.layers.conv1d(self.embedding, pm.filters_num, pm.half_kernel_size*2+1, padding='valid')
            conv = tf.reshape(conv, [-1, pm.filters_num])

        with tf.name_scope('output'):
            # 本模型关键， 合并BiLSTM、CNN结果
            output = tf.concat([conv, output], -1)

            # 建立一个全连接网络
            # 此时output 变为(batch_size, pm.units)
            # 达到分类要求 BMES
            output = tf.layers.dense(output, pm.units)
            # 为该网络包裹一层dropout
            output = tf.contrib.layers.dropout(output, self.keep_pro)

        with tf.name_scope('CRF'):
            self.logits = tf.reshape(output, [-1, shape[1], pm.units])
            self.log_likelihood, self.transition_matrix = crf_log_likelihood(inputs=self.logits,
                                                    tag_indices=self.y_intput, sequence_lengths=self.seq_length)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-self.log_likelihood)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variable = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variable), global_step=self.global_step)

    def feed_data(self, x_batch, y_batch, seq_length, keep_pro):
        return {self.x_input: x_batch,
                self.y_intput: y_batch,
                self.seq_length: seq_length,
                self.keep_pro: keep_pro}

    def test(self, sess, x, y):
        """
        计算当前batch下，损失，用在训练模块中
        :param sess: 提供会话
        :param x: x输入
        :param y: y标签
        :return: loss
        """
        batch = t.next_batch(x, y, batch_size=pm.batch_size)
        loss = 0
        count = 0
        for x_batch, y_batch in batch:
            count += 1
            x_batch, seq_length_x = t.sentence_process(x_batch)
            y_batch, seq_length_y = t.sentence_process(y_batch)
            feed_dict = self.feed_data(x_batch, y_batch, seq_length_x, 1)
            loss += sess.run(self.loss, feed_dict)
        return loss / count

    def predict(self, sess, x_batch):
        """
        预测值，用在预测模块中
        :param sess: 提供会话
        :param x_batch: 一批输入
        :return: 每个字的label
        """
        seq, seq_length = t.sentence_process(x_batch)
        logits, transition_matrix = sess.run([self.logits, self.transition_matrix], feed_dict={
            self.x_input: seq, self.seq_length: seq_length, self.keep_pro: 1.0})

        label = []
        for logit, length in zip(logits, seq_length):
            viterbi_seq, _ = viterbi_decode(logit[:length], transition_matrix)
            label.append(viterbi_seq)
        return label