from lstm import LSTM
import reader
import tensorflow as tf
import numpy as np

LR = 0.001
BATCH_SIZE = 64
TIME_STEPS = 100
INPUT_SIZE = 1
HIDDEN_UNITS = 128  # CELL_SIZE = #hidden_units in the cell


def train(sess, network_input, network_output, model):
    for epoch in range(5):
        total_batch = int(network_input.shape[0] / BATCH_SIZE)
        epoch_cost = 0
        for i in range(total_batch):
            randidx = np.random.randint(int(network_input.shape[0]), size=BATCH_SIZE)
            batch_xs = network_input[randidx, :]
            batch_ys = network_output[randidx, :]
            batch_xs = batch_xs.reshape([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
            print("batch shape: ", batch_xs.shape, batch_ys.shape)
            if i==0:
                feed_dict = {
                    model.x: batch_xs,
                    model.y: batch_ys,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.x: batch_xs,
                    model.y: batch_ys,
                    model.cell_init_state: state,
                }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict
            )

            epoch_cost = epoch_cost + cost
        print("cost: ", epoch_cost)

def pred(sess, feed_dict):
    return sess.run(model, pred, feed_dict=feed_dict)

def main(_):
    # set random seed for comparing the two result calculations
    tf.set_random_seed(1)

    notes = reader.get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = reader.prepare_sequences(notes, n_vocab)

    # model = create_network(network_input, n_vocab)
    # train(model, network_input, network_output)

    model = LSTM(LR, BATCH_SIZE, TIME_STEPS, INPUT_SIZE, HIDDEN_UNITS, n_vocab)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    train(sess, network_input, network_output, model)


    # result = pred(sess, network_input, {})


    sess.close()

if __name__ == '__main__':
    tf.app.run()
