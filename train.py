#-.- coding: utf-8 -.-
import tensorflow as tf
import numpy as np
import os
import time
import shutil
import datetime
import data_helpers
from text_cnn import TextCNN
# Parameters
# ==================================================

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("early_stop_count", 100, "if accuracy not better for this time , stop training")
tf.flags.DEFINE_boolean("tmp_dir", False, "if True, write model result to ./runs/tmp/")
# Misc Parameters
"""
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
"""

tf.flags.DEFINE_string("data_path", "./data/data_server", "dir where training data are")
tf.flags.DEFINE_string("training_info", "default training setting", "any info differing this training from others")
flags = tf.app.flags
FLAGS = flags.FLAGS

train_info_dict={}

# Training
# ==================================================
def train(data_set):
  print(FLAGS.training_info)
  #train_info_dict.append(FLAGS.training_info)
  print("\nParameters:")
  for attr, value in sorted(FLAGS.__flags.iteritems()):
      print("{}={}".format(attr.upper(), value))
      train_info_dict[attr]= value
  print("")
  with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=data_set.x_train.shape[1],
            num_classes=data_set.y.shape[1],
            vocab_size=len(data_set.vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        """
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """
        """
        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        """
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        if FLAGS.tmp_dir: 
          out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs','tmp', timestamp))
        print("Writing to {}\n".format(out_dir))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        model_store_dir = os.path.abspath(os.path.join(out_dir, "model_store"))
        train_info_dict['model_dir']=model_store_dir #记录模型存放位置
        best_checkpoint_dir = os.path.abspath(os.path.join(model_store_dir, "best_checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(model_store_dir):
            os.makedirs(model_store_dir)
        """write info to file"""
        model_info_dir=os.path.join(model_store_dir,'model_info')
        with open(model_info_dir,'w') as fmi:
          for k,v in train_info_dict.items():
            fmi.write('{}={}\n'.format(k,v))
        with open(os.path.join(model_store_dir,'label_inv'),'w') as f:
          for(k,v) in data_set.label_inv.items():
            f.write(" ".join((str(k),v)))
            f.write('\n')
        print("End write label-inv")
        with open(os.path.join(model_store_dir,"vocabulary"),'w') as f:
          for(k,v) in data_set.vocabulary.items():
            f.write(" ".join((k,str(v))))
            f.write('\n')
        print("End write vocabulary")

        saver = tf.train.Saver(tf.all_variables(),keep_checkpoint_every_n_hours=1)
        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            """
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            """
            start_time = time.time()
            _, step, loss, accuracy = sess.run(
                [train_op, global_step,  cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            duration = time.time() - start_time
      
            if step % 100000 == 0:
              num_examples_per_step = FLAGS.batch_size 
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = duration 
      
              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.datetime.now(), step, loss,
                                 examples_per_sec, sec_per_batch))
            if step%FLAGS.checkpoint_every==0:
              print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy

        # Generate batches
        batches = data_helpers.batch_iter(
            zip(data_set.x_train,data_set.y_train), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        max_accuracy=0.0
        max_path=''
        max_count=0
        for batch in batches:
            if len(batch)==0:
              continue
            x_batch, y_batch = zip(*batch)
            current_step = tf.train.global_step(sess, global_step)
            train_step(x_batch, y_batch)
    
            dev_accuray=0;
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_accuracy=dev_step(data_set.x_dev, data_set.y_dev)
                # writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                print(max_accuracy)
                if dev_accuracy >max_accuracy:
                  max_accuracy = dev_accuracy
                  max_count=1
                  max_path=path
                  shutil.copy(path,best_checkpoint_dir)
                else:
                  max_count +=1
                if max_count > FLAGS.early_stop_count:
                  print("early stop and best accuracy model at "+max_path)
                  return 
        print('end traing because hitting num_epochs {}'.format(FLAGS.num_epochs))


if __name__ == '__main__':
  data_set = data_helpers.DataSet(FLAGS.data_path)
  train(data_set)
