import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import os
import h5py
import utils.image_processing as ip
import utils.plot_functions as pf

def get_dataset(num_images, out_shape):
  img_filename = (os.path.expanduser("~")
    +"/Work/Datasets/vanHateren/img/images_curated.h5")
  full_img_data = extract_images(img_filename, num_images)
  full_img_data = ip.downsample_data(full_img_data, factor=[1, 0.5, 0.5],
    order=2)
  full_img_data = ip.standardize_data(full_img_data)
  dataset = ip.extract_patches(full_img_data, out_shape, True, 1e-6)
  return dataset

def extract_images(filename, num_images=50):
  with h5py.File(filename, "r") as f:
    full_img_data = np.array(f["van_hateren_good"], dtype=np.float32)
    im_keep_idx = np.random.choice(full_img_data.shape[0], num_images,
      replace=False)
    return full_img_data[im_keep_idx, ...]

## Model params
out_dir = os.path.expanduser("~")+"/Work/DensityLearner/outputs/"
update_interval = 500
device = "/cpu:0"
num_batches = 30000
batch_size = 100
learning_rate = 0.01

## Data params
num_images = 50
epoch_size = int(1e6)
patch_edge_size = 20

## Calculated params
num_pixels = int(patch_edge_size**2)
dataset_shape = [int(val)
  for val in [epoch_size, num_pixels]]
num_neurons = num_pixels # Complete ICA
a_shape = [num_pixels, num_neurons]

## Graph construction
graph = tf.Graph()
with tf.device(device):
  with graph.as_default():
    with tf.name_scope("placeholders") as scope:
      x = tf.placeholder(
        tf.float32, shape=[batch_size, patch_edge_size**2], name="input_data")

    with tf.name_scope("step_counter") as scope:
      global_step = tf.Variable(0, trainable=False, name="global_step")

    with tf.variable_scope("weights") as scope:
      Q, R = np.linalg.qr(np.random.standard_normal(a_shape))
      a = tf.get_variable(name="a", dtype=tf.float32,
        initializer=Q.astype(np.float32), trainable=True)

    with tf.name_scope("inference") as scope:
      u = tf.matmul(x, tf.matrix_inverse(a,
        name="a_inverse"), name="coefficients")
      z = tf.sign(u) # laplacian dist

    with tf.name_scope("optimizers") as scope:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate,
        name="a_optimizer")
      z_u_avg = tf.divide(tf.matmul(tf.transpose(u), z),
        tf.to_float(batch_size), name="avg_samples")
      gradient = -tf.subtract(tf.matmul(z_u_avg, a), a,
        name="a_gradient")
      update_weights = optimizer.apply_gradients([(gradient, a)],
        global_step=global_step)

    full_saver = tf.train.Saver(var_list=[a], max_to_keep=2)

    with tf.name_scope("summaries") as scope:
      #tf.summary.image("input", tf.reshape(x, [batch_size, patch_edge_size,
      #  patch_edge_size, 1]))
      #tf.summary.image("weights", tf.reshape(tf.transpose(a), [num_neurons,
      #  patch_edge_size, patch_edge_size, 1]))
      tf.summary.histogram("u", u)
      tf.summary.histogram("z", z)
      tf.summary.histogram("a", a)
      tf.summary.histogram("a_gradient", gradient)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(out_dir, graph)

    with tf.name_scope("initialization") as scope:
      init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

print("Getting data...")
dataset = get_dataset(num_images, dataset_shape)

print("Training model...")
## Train the model
with tf.Session(graph=graph) as sess:
  sess.run(init_op,
    feed_dict={x:np.zeros([batch_size, num_pixels], dtype=np.float32)})

  epoch_order = np.random.permutation(epoch_size)
  curr_epoch_idx = 0
  for b_step in range(num_batches):
    if curr_epoch_idx + batch_size > epoch_size:
      start = 0
      curr_epoch_idx = 0
      epoch_order = np.random.permutation(epoch_size)
    else:
      start = curr_epoch_idx
    curr_epoch_idx += batch_size
    data_batch = dataset[epoch_order[start:curr_epoch_idx], ...]

    feed_dict = {x:data_batch}
    sess.run(update_weights, feed_dict)

    step = sess.run(global_step)
    if step % update_interval == 0:
      summary = sess.run(merged_summaries, feed_dict)
      train_writer.add_summary(summary, step)
      full_saver.save(sess, save_path=out_dir+"ica_chk", global_step=global_step)
      weights = sess.run(a, feed_dict)
      pf.save_data_tiled(weights.reshape(num_neurons,
        patch_edge_size, patch_edge_size), normalize=False,
        title="A matrix at step "+str(step),
        save_filename=out_dir+"a_weights_"+str(step)+".png")
      print("step "+str(step))

import IPython; IPython.embed(); raise SystemExit
