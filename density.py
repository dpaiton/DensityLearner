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
checkpoint_location = os.path.join(out_dir,"ica_chk-30000")
log_interval = 1000
checkpoint_interval = 10000
device = "/cpu:0"
num_batches = 30000
batch_size = 500 #500 for K&L
learning_rate = 0.01 #0.005 for K&L
v_step_size = 0.02 #K&L
eps = 1e-12
v_sparse_mult = np.sqrt(2.0).astype(np.float32) # From K&L

## Data params
num_images = 50
epoch_size = int(1e6)
patch_edge_size = 20
num_v = 20 #4 for K&L

## Calculated params
num_pixels = int(patch_edge_size**2)
dataset_shape = [int(val)
  for val in [epoch_size, num_pixels]]
num_neurons = num_pixels # Complete ICA
a_shape = [num_pixels, num_neurons]
b_shape = [num_neurons, num_v]

## Graph construction
graph = tf.Graph()
with tf.device(device):
  with graph.as_default():
    with tf.name_scope("placeholders") as scope:
      x = tf.placeholder(
        tf.float32, shape=[batch_size, patch_edge_size**2], name="input_data")

    with tf.name_scope("step_counter") as scope:
      global_step = tf.Variable(0, trainable=False, name="global_step")

    with tf.name_scope("constants") as scope:
      small_v = tf.multiply(0.1, tf.ones(
        shape=tf.stack([batch_size, num_v]),
        dtype=tf.float32, name="v_ones"), name="small_v_init")

    with tf.variable_scope("weights") as scope:
      Q, R = np.linalg.qr(np.random.standard_normal(a_shape))
      a = tf.get_variable(name="a", dtype=tf.float32,
        initializer=Q.astype(np.float32), trainable=True)
      b_init = tf.truncated_normal(b_shape, mean=0.0,
        stddev=1.0, dtype=tf.float32, name="b_init")
      b = tf.get_variable(name="b", dtype=tf.float32,
        initializer=b_init, trainable=True)

    with tf.name_scope("norm_weights") as scope:
      l2_norm_b = b.assign(tf.nn.l2_normalize(b, dim=0, epsilon=eps,
        name="row_l2_norm"))
      norm_weights = tf.group(l2_norm_b, name="l2_normalization")

    with tf.name_scope("inference") as scope:
      u = tf.matmul(x, tf.matrix_inverse(a, name="a_inverse"),
        name="coefficients")
      z = tf.sign(u) # laplacian dist
      v = tf.Variable(small_v, trainable=False, name="v")
      sigma = tf.exp(tf.matmul(v, tf.transpose(b)), name="sigma")

    with tf.name_scope("output") as scope:
      with tf.name_scope("layer_0_estimate"):
        z_recon = tf.divide(tf.abs(z), sigma)

    with tf.name_scope("optimizers") as scope:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate,
        name="optimizer")
      op1 = tf.subtract(1.0, z_recon)
      op2 = tf.matmul(tf.transpose(op1), v)
      b_gradient = tf.divide(op2, 2.0*float(batch_size))
      update_weights = optimizer.apply_gradients([(b_gradient, b)],
        global_step=global_step)

    with tf.name_scope("update_v") as scope:
      z_recon_err = tf.subtract(z_recon, 1.0)
      projected_err = tf.matmul(z_recon_err, b)
      dedv = tf.add(projected_err, tf.multiply(v_sparse_mult,
        tf.sign(v)), name="dv")
      dv = -v_step_size * dedv
      step_v = v.assign_add(dv)

    ica_saver = tf.train.Saver(var_list=[a], max_to_keep=2)
    density_saver = tf.train.Saver(var_list=[b], max_to_keep=2)
    full_saver = tf.train.Saver(var_list=[a, b], max_to_keep=2)

    with tf.name_scope("summaries") as scope:
      #tf.summary.image("input", tf.reshape(x, [batch_size, patch_edge_size,
      #  patch_edge_size, 1]))
      #tf.summary.image("weights", tf.reshape(tf.transpose(a), [num_neurons,
      #  patch_edge_size, patch_edge_size, 1]))
      tf.summary.histogram("u", u)
      tf.summary.histogram("z", z)
      tf.summary.histogram("a", a)
      tf.summary.histogram("b", b)
      tf.summary.histogram("sigma", sigma)
      tf.summary.scalar("avg_z_recon_err", tf.reduce_mean(z_recon_err))
      tf.summary.histogram("b_gradient", b_gradient)

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

  ica_saver.restore(sess, checkpoint_location)

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
    sess.run(norm_weights, feed_dict)

    step = sess.run(global_step)
    if step % log_interval == 0:
      summary = sess.run(merged_summaries, feed_dict)
      train_writer.add_summary(summary, step)
      b_weights = sess.run(b, feed_dict)
      pf.save_data_tiled(b_weights.reshape(num_v,
        int(np.sqrt(num_neurons)), int(np.sqrt(num_neurons))), normalize=False,
        title="A matrix at step "+str(step),
        save_filename=out_dir+"b_weights_"+str(step)+".png")
      print("step "+str(step))
    if step % checkpoint_interval == 0:
      full_saver.save(sess, save_path=out_dir+"density_chk", global_step=global_step)

import IPython; IPython.embed(); raise SystemExit
