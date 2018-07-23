import tensorflow as tf

class LSTM:
    def __init__(self, lr, batch_size, n_steps, n_inputs, n_hidden_units, n_classes):
        # hyperparameters
        self._lr = lr
        self._batch_size = batch_size

        self._n_steps = n_steps  # time steps
        self._n_inputs = n_inputs  # num of features
        self._n_hidden_units = n_hidden_units  # neurons in hidden layer
        self._n_classes = n_classes # num of unique note

        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="xs")
            self.y = tf.placeholder(tf.float32, [None, n_classes], name="ys")

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell()

        with tf.variable_scope('out_hidden'):
            self.add_output_layer()

        with tf.name_scope('cost'):
            self.compute_cost()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)
    
    def add_input_layer(self):
        l_in_x = tf.reshape(self.x, [-1, self._n_inputs], name='2_2D')  # (batch*n_step, in_size)
        # Ws (_n_inputs, _n_hidden_units)
        Ws_in = self._weight_variable([self._n_inputs, self._n_hidden_units])

        # bs (_n_hidden_units, )
        bs_in = self._bias_variable([self._n_hidden_units,])

        # l_in_y = (batch*n_step, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in

        self._l_in_y = tf.reshape(l_in_y, [-1, self._n_steps, self._n_hidden_units], name='2_3D')


    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self._batch_size, dtype=tf.float32)

        # rnn loop
        self._cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self._l_in_y, initial_state=self.cell_init_state,time_major=False
        )


    def add_output_layer(self):
        l_out_x = tf.unstack(tf.transpose(self._cell_outputs, [1, 0, 2]))
        Ws_out = self._weight_variable([self._n_hidden_units, self._n_classes])
        bs_out = self._bias_variable([self._n_classes,])

        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.nn.softmax(tf.matmul(l_out_x[-1], Ws_out) + bs_out)


    def compute_cost(self):
        losses = tf.keras.backend.categorical_crossentropy(self.y, self.pred)
        # self.cost = tf.reduce_mean(losses)

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self._batch_size,
                name='average_cost'
            )
            # tf.summary.scalar('cost', self.cost)
        # tf.summary.scalar('cost', self.cost)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
