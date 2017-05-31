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
project_dir = os.path.expanduser("~")+"/Work/DensityLearner/"
out_dir = os.path.join(project_dir, "test_density_outputs/")
checkpoint_location = os.path.join(project_dir, "ica_outputs/ica_chk-30000")
log_interval = 10
checkpoint_interval = 10000
device = "/cpu:0"
num_batches = 1000#10000
batch_size = 300
b_learning_rate = 0.01
v_step_size = 0.01
num_v_steps = 50
num_v = 100
eps = 1e-12

## Data params
num_images = 10#50
epoch_size = num_batches * batch_size
patch_edge_size = 20

## Calculated params
num_pixels = int(patch_edge_size**2.0)
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
      x = tf.placeholder(tf.float32,
        shape=[batch_size, patch_edge_size**2], name="input_data")

    with tf.name_scope("step_counter") as scope:
      global_step = tf.Variable(0, trainable=False, name="global_step")

    with tf.name_scope("constants") as scope:
      small_v = tf.multiply(0.1, tf.ones(
        shape=tf.stack([batch_size, num_v]),
        dtype=tf.float32, name="v_ones"), name="small_v_init")
      v_zeros = tf.zeros(shape=tf.stack([batch_size, num_v]), dtype=tf.float32,
        name="v_zeros")

    with tf.variable_scope("weights") as scope:
      Q, R = np.linalg.qr(np.random.standard_normal(a_shape))
      a = tf.get_variable(name="a", dtype=tf.float32,
        initializer=Q.astype(np.float32), trainable=True)
      #b_init = tf.nn.l2_normalize(tf.truncated_normal(b_shape, mean=0.0,
      #  stddev=0.1, dtype=tf.float32, name="b_init"), dim=0, epsilon=eps,
      #  name="normed_b_init")
      b_init = tf.random_uniform(b_shape, minval=-0.1, maxval=0.1,
        dtype=tf.float32, seed=None, name="b_init")
      b = tf.get_variable(name="b", dtype=tf.float32,
        initializer=b_init, trainable=True)

    with tf.name_scope("norm_weights") as scope:
      l2_norm_b = b.assign(tf.nn.l2_normalize(b, dim=0, epsilon=eps,
        name="row_l2_norm"))
      l1_norm_b = b.assign(tf.divide(b, tf.add(eps, tf.reduce_sum(tf.abs(b),
        axis=0)), name="row_l1_norm"))
      #norm_weights = tf.group(l2_norm_b, name="l2_normalization")
      norm_weights = tf.group(l1_norm_b, name="l1_normalization")

    with tf.name_scope("inference") as scope:
      u = tf.matmul(x, tf.matrix_inverse(a, name="a_inverse"),
        name="coefficients")
      v = tf.Variable(v_zeros, trainable=False, name="v")
      sigma = tf.exp(tf.matmul(v, tf.transpose(b)), name="sigma")
      half_sigma = tf.exp(tf.multiply(tf.matmul(v, tf.transpose(b)), 2.0),
        name="half_sigma")
      # Z from ICA is replaced with p(s|B,v) from density model
      z = tf.divide(tf.exp(-tf.divide(tf.multiply(tf.sqrt(2.0), tf.abs(u)),
        tf.sqrt(sigma))), tf.sqrt(tf.multiply(2.0, sigma)))
      u_recon = tf.divide(tf.multiply(tf.sqrt(2.0), u), half_sigma)

    with tf.name_scope("update_v") as scope:
      u_recon_err = tf.subtract(tf.abs(u_recon), 1.0)
      projected_err = tf.matmul(u_recon_err, b)
      dedv = tf.subtract(projected_err, tf.multiply(tf.sqrt(2.0),
        tf.sign(v)), name="dedv")
      #aprox_hessian = -tf.matmul(tf.abs(u_recon), tf.square(b))
      #v_gradient_scale = tf.divide(v_step_size,
      #  tf.add(tf.reduce_mean(aprox_hessian, axis=0), 1e-12))
      v_gradient_scale = tf.constant(v_step_size)
      dv = tf.multiply(v_gradient_scale, dedv)
      step_v = v.assign_add(dv)
      reset_v = v.assign(v_zeros)

    with tf.name_scope("optimizers") as scope:
      optimizer = tf.train.GradientDescentOptimizer(b_learning_rate,
        name="optimizer")
      op1 = tf.subtract(u_recon, 1.0)
      op2 = tf.matmul(tf.transpose(op1), v)
      b_gradient = tf.divide(op2, tf.to_float(batch_size))
      update_weights = optimizer.apply_gradients([(b_gradient, b)],
        global_step=global_step)

    with tf.name_scope("likelihood") as scope:
      #likelihood = tf.subtract(tf.add(-tf.log(tf.abs(tf.matrix_determinant(a))),
      #  tf.reduce_mean(tf.reduce_sum(tf.subtract(tf.divide(-tf.log(sigma),
      #  2.0), u_recon), axis=1), axis=0)),
      #  tf.reduce_mean(tf.reduce_sum(tf.abs(v), axis=1), axis=0))
      likelihood = tf.subtract(
        tf.reduce_mean(tf.reduce_sum(tf.subtract(tf.divide(-tf.log(sigma),
        2.0), u_recon), axis=1), axis=0),
        tf.reduce_mean(tf.reduce_sum(tf.abs(v), axis=1), axis=0))

    ica_saver = tf.train.Saver(var_list=[a], max_to_keep=2)
    density_saver = tf.train.Saver(var_list=[b], max_to_keep=2)
    full_saver = tf.train.Saver(var_list=[a, b], max_to_keep=2)

    with tf.name_scope("summaries") as scope:
      #tf.summary.image("input", tf.reshape(x, [batch_size, patch_edge_size,
      #  patch_edge_size, 1]))
      #tf.summary.image("weights", tf.reshape(tf.transpose(a), [num_neurons,
      #  patch_edge_size, patch_edge_size, 1]))
      tf.summary.histogram("v_step_size", v_gradient_scale)
      tf.summary.histogram("u", u)
      tf.summary.histogram("z", z)
      tf.summary.histogram("v", v)
      tf.summary.histogram("a", a)
      tf.summary.histogram("b", b)
      tf.summary.histogram("sigma", sigma)
      tf.summary.histogram("likelihood", likelihood)
      tf.summary.scalar("avg_u_recon_err", tf.reduce_mean(u_recon_err))
      tf.summary.histogram("b_gradient", b_gradient)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(out_dir, graph)

    with tf.name_scope("initialization") as scope:
      init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

print("Getting data...")
dataset = get_dataset(num_images, dataset_shape)

print("Training model...")
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
## Train the model
step = 0
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
    sess.run(reset_v, feed_dict)
    for v_step in range(num_v_steps):
      [_, v_eval, v_gradient_eval] = sess.run([step_v, v, v_gradient_scale], feed_dict)
      if np.any(np.isnan(v_eval)):
        print("V has NANs")
        import IPython; IPython.embed(); raise SystemExit
      #pf.save_activity_hist(v_gradient_eval, num_bins="auto",
      #  title="V Gradient Histogram "+str(step)+"_"+str(v_step).zfill(3),
      #  save_filename=out_dir+"v_grad_hist_"+str(step)+"_"+str(v_step).zfill(3))
      #pf.save_activity_hist(v_eval, num_bins="auto",
      #  title="V Activity Histogram "+str(step)+"_"+str(v_step).zfill(3),
      #  save_filename=out_dir+"v_act_hist_"+str(step)+"_"+str(v_step).zfill(3))
      #v_norm = np.linalg.norm(v_eval, axis=1)
      #pf.save_activity_hist(v_norm, num_bins="auto",
      #  title="V l2-Norm Histogram "+str(step)+"_"+str(v_step).zfill(3),
      #  save_filename=out_dir+"v_norm_hist_"+str(step)+"_"+str(v_step).zfill(3))
    sess.run(update_weights, feed_dict)
    sess.run(norm_weights, feed_dict)

    step = sess.run(global_step)
    if step % log_interval == 0:
      summary = sess.run(merged_summaries, feed_dict)
      train_writer.add_summary(summary, step)
      [b_eval, b_grad_eval, likelihood_eval] = sess.run([b, b_gradient,
        likelihood], feed_dict)
      pf.save_data_tiled(b_eval.reshape(num_v,
        int(np.sqrt(num_neurons)), int(np.sqrt(num_neurons))), normalize=False,
        title="B matrix at step "+str(step),
        save_filename=out_dir+"b_weights_"+str(step).zfill(4)+".png")
      pf.save_bar(np.linalg.norm(b_eval, axis=1, keepdims=False),
        num_xticks=5, title="B l2 norm",
        save_filename=(out_dir+"b_norm_"+str(step).zfill(4)+".png"),
        xlabel="Basis Index", ylabel="L2 Norm")
      print("step "+str(step).zfill(4)+"\tlikelihood "+str(likelihood_eval))
    if step % checkpoint_interval == 0:
      full_saver.save(sess, save_path=out_dir+"density_chk",
        global_step=global_step)

import IPython; IPython.embed(); raise SystemExit
