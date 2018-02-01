from __future__ import division, print_function, unicode_literals
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import BatchDatsetReader as dataset
import read_MITSceneParsingData as scene_parsing
import datetime


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e7 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 56
PROCESSED_IMAGE_SIZE = 56

tf.reset_default_graph()

np.random.seed(42)
tf.set_random_seed(42)

X = tf.placeholder(shape=[None,PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE, 3], dtype=tf.float32, name="cn_input_image")


caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 20 * 20  # 1152 primary capsules
caps1_n_dims = 8

conv1_params = {
  "filters": 256,
  "kernel_size": 9,
  "strides": 1,
  "padding": "valid",
  "activation": tf.nn.relu,
}

conv2_params = {
  "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
  "kernel_size": 9,
  "strides": 2,
  "padding": "valid",
  "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="cn_conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="cn_conv2", **conv2_params)



fc_input = tf.reshape(conv2,
                         [-1, caps1_n_caps * caps1_n_dims],
                         name="cn_fc1_input")


n_hidden1 = 512
n_output = caps1_n_caps * caps1_n_dims

hidden1 = tf.layers.dense(fc_input, n_hidden1,
                            activation=tf.nn.relu, name = "cn_fc1_hidden")


fc_output = tf.layers.dense(hidden1, caps1_n_caps * caps1_n_dims,
                                   activation=tf.nn.relu, name = "cn_fc_output")

caps1_input = tf.reshape(fc_output,
                         [-1, 20, 20,256], name = "cn_caps1_input")

print(caps1_input)

caps1_raw = tf.reshape(caps1_input, [-1, caps1_n_caps, caps1_n_dims],
                     name="cn_caps1_raw")


def squash(s, axis=-1, epsilon=1e-7, name=None):
  with tf.name_scope(name, default_name="cn"):
      squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                   keep_dims=True)
      safe_norm = tf.sqrt(squared_norm + epsilon)
      squash_factor = squared_norm / (1. + squared_norm)
      unit_vector = s / safe_norm
      return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="cn_caps1_output")





caps2_n_caps = 4
caps2_n_dims = 151

init_sigma = 0.01

W_init = tf.random_normal(
  shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
W = tf.Variable(W_init, name="cn_W")

W_r_init = tf.random_normal(
  shape=(1, caps1_n_caps, caps2_n_caps, caps1_n_dims, caps2_n_dims),
  stddev=init_sigma, dtype=tf.float32, name="cn_W_initr")
W_r = tf.Variable(W_r_init, name="cn_Wr")

W_sum_init = tf.random_normal(
  shape=(1, caps1_n_caps, caps2_n_caps, 1, 1),
  stddev=init_sigma, dtype=tf.float32, name="cn_W_init_sum")
w_sums = tf.Variable(W_sum_init, name="cn_W_sum")


batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="cn_W_tiled")
W_tiled_r = tf.tile(W_r, [batch_size, 1, 1, 1, 1], name="cn_W_tiled_r")


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                     name="cn_caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                 name="cn_caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                           name="cn_caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                          name="cn_caps2_predicted")







raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                     dtype=np.float32, name="cn_raw_weights")

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="cn_routing_weights")

weighted_predictions_init = tf.multiply(routing_weights, caps2_predicted,
                                 name="cn_weighted_predictions")
weighted_predictions = tf.multiply(w_sums, weighted_predictions_init,
                                 name="cn_weighted_predictions_sum")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                           name="cn_weighted_sum")

caps2_output_round_1_init = squash(weighted_sum, axis=-2,
                            name="cn_caps2_output_round_1_init")




################################################

fc_input_2 = tf.reshape(caps2_output_round_1_init,
                         [-1, caps2_n_caps * caps2_n_dims],
                         name="cn_fc2_input")


n_hidden2 = caps2_n_caps * caps2_n_dims


fc_output2 = tf.layers.dense(fc_input_2, n_hidden2,
                            activation=tf.nn.relu,
                            name="cn_fc2_output")


caps2_output_round_1 = squash(tf.add(caps2_output_round_1_init, tf.reshape(fc_output2,
                         [-1, 1,4, 151,1],
                         name="cn_caps2_output_round_1")),axis=-2)

##################################################################

caps2_output_round_1_tiled = tf.tile(
  caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
  name="cn_caps2_output_round_1_tiled")

caps1_output_tile_modified = tf.matmul(W_tiled_r, caps2_output_round_1_tiled,
                          name="cn_caps1_output_tile_modified")


caps2_predicted_modified_recude_dim = tf.reduce_sum(caps1_output_tile_modified, axis=2, keep_dims=True, name = "cn_caps2_predicted_modified_recude_dim")
                           

caps2_predicted_added = tf.add(caps2_predicted_modified_recude_dim , caps1_output_tile, name = "cn_caps2_predicted_added")

caps1_output_tiled_modified = tf.tile(caps2_predicted_added, [1, 1, caps2_n_caps, 1, 1], name = "cn_caps1_output_tiled_modified")

caps2_predicted_modified = tf.matmul(W_tiled, caps1_output_tiled_modified, name = "cn_caps2_predicted_modified")

agreement = tf.matmul(caps2_predicted_modified, caps2_output_round_1_tiled,
                    transpose_a=True, name="cn_agreement")


raw_weights_round_2 = tf.add(raw_weights, agreement,
                           name="cn_raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                      dim=2,
                                      name="cn_routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                         caps2_predicted_modified,
                                         name="cn_weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                   axis=1, keep_dims=True,
                                   name="cn_weighted_sum_round_2")
caps2_output_round_2_init = squash(weighted_sum_round_2,
                            axis=-2,
                            name="cn_caps2_output_round_2")

######################################################

fc_input_3 = tf.reshape(caps2_output_round_2_init,
                         [-1, caps2_n_caps * caps2_n_dims],
                         name="cn_fc3_input")


n_hidden3 = caps2_n_caps * caps2_n_dims


fc_output3 = tf.layers.dense(fc_input_3, n_hidden3,
                            activation=tf.nn.relu,
                            name="cn_hidden3")


caps2_output_round_2 = tf.add(caps2_output_round_2_init, tf.reshape(fc_output3,
                         [-1, 1,4, 151,1],
                         name="cn_caps2_output_round_2"))




###################################################


caps2_output = caps2_output_round_2


decoder_input = tf.reshape(caps2_output,
                         [-1, caps2_n_caps * caps2_n_dims],
                         name="cn_decoder_input")


n_hidden1_decoder = 512
n_hidden2_decoder = 1024
n_output_decoder = 56 * 56


with tf.name_scope("decoder"):
  hidden1_de = tf.layers.dense(decoder_input, n_hidden1_decoder,
                            activation=tf.nn.relu, name = "cn_hidden1_de")
  hidden2_de = tf.layers.dense(hidden1_de, n_hidden2_decoder,
                            activation=tf.nn.relu, name = "cn_hidden2_de")
  decoder_output_init = tf.layers.dense(hidden2_de, n_output_decoder,
                                   activation=tf.nn.relu, name = "cn_decoder_output_init")


decoder_output = tf.round(decoder_output_init, name = "cn_decoder_output")
decoder_annotation = tf.placeholder(tf.float32, shape=[None, 56, 56, 1], name="cn_decoder_annotation")
annot_flat = tf.reshape(decoder_annotation, [-1, n_output_decoder], name="cn_annot_flat")
squared_difference = tf.square(annot_flat - decoder_output,
                             name="cn_squared_difference")

decoder_loss = tf.reduce_mean(squared_difference,
                                  name="cn_reconstruction_loss")





y_pred = tf.cast(tf.squeeze(tf.to_int32(caps2_output > 0.5),axis=[1,4]),tf.float32, name = "cn_y_pred")

print(y_pred)


m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5



annotation = tf.placeholder(tf.float32, shape=[None, 4, 151, 1], name="cn_annotation")
shaped_annotations = tf.squeeze(annotation, squeeze_dims=[3], name="cn_shaped_annotations")
#shaped_caps2_output = tf.nn.softmax(tf.squeeze(caps2_output, squeeze_dims=[1,4]),dim = 2)

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output), name="cn_present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 4, 151), name="cn_present_error")


absent_error_raw = tf.square(tf.maximum(0., caps2_output - m_minus),name="cn_absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1,4, 151), name="cn_absent_error")


L = tf.add(tf.multiply(shaped_annotations,present_error), lambda_ * tf.multiply((1.0 - shaped_annotations), absent_error),name="cn_L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="cn_margin_loss")

alpha = 0.05

loss = tf.add(margin_loss, alpha * decoder_loss, name = "cn_loss")



best_loss_val = np.infty
correct = tf.equal(shaped_annotations, y_pred, name="cn_correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="cn_accuracy")

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="cn_training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver()


train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)

image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
train_dataset_reader = dataset.BatchDatset(train_records, image_options)
validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)


sess = tf.Session()

n_epochs = 10
batch_size = 50
restore_checkpoint = True


recall = tf.metrics.recall(labels = shaped_annotations, predictions = y_pred, name="cn_recall")
precision = tf.metrics.precision(labels = shaped_annotations, predictions = y_pred, name = "cn_precision")


checkpoint_path = "./my_capsule_network"

#151 classes


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#saver.restore(sess, checkpoint_path)
epoch = 0
for itr in xrange(MAX_ITERATION):
	loss_vals = []
	acc_vals = []
	recall_vals = []
	precision_vals = []
# for val_itr in range(1,1000):
# 	X_batch, y_batch = validation_dataset_reader.next_batch(FLAGS.batch_size)
# 	loss_val, acc_val, recall_val, precision_val = sess.run([loss,accuracy, recall, precision], feed_dict={X: X_batch.reshape([-1, PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE, 3]),annotation: y_batch})
# 	loss_vals.append(loss_val)
# 	acc_vals.append(acc_val)
# 	recall_vals.append(recall_val)
# 	precision_vals.append(precision_val)
# 	print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
# 			val_itr, 1000,
# 			val_itr * 100 / 1000),
# 			end=" " * 10)
	
# loss_val = np.mean(loss_vals)
# acc_val = np.mean(acc_vals)
# recall_val = np.mean(recall_vals)
# precision_val = np.mean(precision_vals)

# print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{} Recall: {:.6f}   Precision: {:.6f}".format(
# 		epoch, acc_val * 100, loss_val,
# 		" (improved)" if loss_val < best_loss_val else "", recall_val, precision_val))





	X_batch, y_batch, y_annotation = train_dataset_reader.next_batch(FLAGS.batch_size)
	_,loss_train = sess.run([training_op, loss],feed_dict={X: X_batch.reshape([-1, PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE, 3]),annotation: y_batch, decoder_annotation:y_annotation})

	if (itr % 10) == 0:
	# 	print("\rIteration: {} ({:.1f}%)  Loss: {:.5f}".format(itr,itr * 100 / MAX_ITERATION,loss_train),end="")
		print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
	                    itr, 10000,
	                    itr * 100 / 10000,
	                    loss_train),
	                end="")
	if (itr % 10000) == 0:
		loss_vals = []
		acc_vals = []
		recall_vals = []
		precision_vals = []
		decoder_vals = []
		for val_itr in range(1,1000):
			X_batch, y_batch, y_annotation = validation_dataset_reader.next_batch(FLAGS.batch_size)
			loss_val, acc_val, recall_val, precision_val, decoder_val = sess.run([loss,accuracy, recall, precision, decoder_loss], feed_dict={X: X_batch.reshape([-1, PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE, 3]),annotation: y_batch, decoder_annotation:y_annotation})
			loss_vals.append(loss_val)
			acc_vals.append(acc_val)
			recall_vals.append(recall_val)
			decoder_vals.append(decoder_val)
			precision_vals.append(precision_val)
			print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
					val_itr, 1000,
					val_itr * 100 / 1000),
					end=" " * 10)
			
		loss_val = np.mean(loss_vals)
		acc_val = np.mean(acc_vals)
		recall_val = np.mean(recall_vals)
		precision_val = np.mean(precision_vals)
		decoder_val = np.mean(decoder_vals)
		epoch = epoch + 1
		print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{} Recall: {:.6f}   Precision: {:.6f} Decoder_loss: {:.6f}".format(
				epoch, acc_val * 100, loss_val,
				" (improved)" if loss_val < best_loss_val else "", recall_val, precision_val, decoder_val))
		if loss_val < best_loss_val:
			save_path = saver.save(sess, checkpoint_path)
			best_loss_val = loss_val
		#save_path = saver.save(sess, checkpoint_path)
