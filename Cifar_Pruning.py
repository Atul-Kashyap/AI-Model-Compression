from utils import *

get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'notebook')



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



epochs = 50
epochs_prune = 5
batch_size = 200
model_dir = 'Models'
model_path_unpruned = os.path.join(model_dir,'Unpruned_{}.ckpt')
model_path_pruned = os.path.join(model_dir,'Pruned_{}.ckpt')
NUM_CLASS = 10
batch_size_test = batch_size

##Data Processing

#Load Dataset
X_train0, y_train = prepare_dataset(data_dir, 'train')
X_test0, y_test = prepare_dataset(data_dir, 'test')
t = int(time.time())

#Normalizing
mean = np.mean(X_train0,axis=(0,1,2,3))
std = np.std(X_train0,axis=(0,1,2,3))
np.save('mean',mean)
np.save('std',std)
X_train = z_normalization(X_train0, mean, std)
X_test = z_normalization(X_test0, mean, std)

#Labels to binary
y_train_binary = keras.utils.to_categorical(y_train,num_classes)
y_test_binary = keras.utils.to_categorical(y_test,num_classes)

#Calculate number of batches
batches = int(len(X_train) / batch_size)
batches_test = int(len(X_test) / batch_size)
print(batches)
print(batches_test)



# Create tf model graph
tf.reset_default_graph()
image = tf.placeholder(name='images', dtype=tf.float32, shape=[None, 32, 32, 3])
label = tf.placeholder(name='fine_labels', dtype=tf.int32, shape=[None, 10])
logits = tf_fcn_model(image)

# Create global step variable (needed for pruning)
global_step = tf.train.get_or_create_global_step()
reset_global_step_op = tf.assign(global_step, 0)

# Loss function
loss = tf.losses.softmax_cross_entropy(label, logits)

# Training op, the global step is critical here, make sure it matches the one used in pruning later
# running this operation increments the global_step
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)

# Accuracy ops
#prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=tf.argmax(label, 1), k=5), tf.float32))

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(max_to_keep=100)



#Training an Unpruned Model

with tf.Session() as sess:

    acc = 0
    # Global initializer
    sess.run(tf.global_variables_initializer())

    # Training
    for epoch in range(epochs):
        for batch in range(batches):
            batch_xs, batch_ys = sample_batch(X_train, y_train_binary, batch_size)
            sess.run(train_op, feed_dict={image: batch_xs, label: batch_ys})

        # Calculate Test Accuracy every 2 epochs
        if (epoch+1) % 2 == 0:
            acc_print = 0
            acc_print_5 = 0
            for index, offset in enumerate(range(0, X_test.shape[0], batch_size)):
                batch_xt = np.array(X_test[offset: offset + batch_size,:])
                batch_yt = np.array(y_test_binary[offset: offset +  batch_size])
                acc_print += sess.run(accuracy, feed_dict={image: batch_xt, label: batch_yt})
                acc_print_5 += sess.run(accuracy_5, feed_dict={image: batch_xt, label: batch_yt})
            acc_print = acc_print/batches_test
            acc_print_5 = acc_print_5/batches_test
            print("Un-pruned model epoch %d test accuracy %g" % (epoch+1, acc_print))
            print("Un-pruned model epoch %d test top5-accuracy %g" % (epoch+1, acc_print_5))

            # Saves the best
            if acc_print>acc:
                acc =acc_print
                print('saving')
                saver.save(sess, model_path_unpruned.format('best'))
        print(epoch+1)
    # Saves the final model
    saver.save(sess, model_path_unpruned.format('final'))


#Loading a Saved Un-Pruned Model and prints evaluation metrics
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path_unpruned.format('best'))
        acc_unpruned = 0
        acc_unpruned_5 = 0
        for index, offset in enumerate(range(0, X_test.shape[0], batch_size)):
            batch_xt = np.array(X_test[offset: offset + batch_size,:])
            batch_yt = np.array(y_test_binary[offset: offset +  batch_size])
            acc_unpruned += sess.run(accuracy, feed_dict={image: batch_xt, label: batch_yt})
            acc_unpruned_5 += sess.run(accuracy_5, feed_dict={image: batch_xt, label: batch_yt})
        acc_unpruned = acc_unpruned/batches_test
        acc_unpruned_5 = acc_unpruned_5/batches_test
        print("Un-pruned model epoch %d test accuracy %g" % (1, acc_unpruned))
        print("Un-pruned model epoch %d test top5-accuracy %g" % (1, acc_unpruned_5))
        print("Sparsity of layers (should be 0)", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))


## Pruning the model for different sparsities and evaluating the performance

sparsity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

acc = []
acc_5 = []
for s in sparsity:
    prune_op = set_prune_params(s)
    print('Target sparsity is {}'.format(s))

    with tf.Session() as sess:
        # Resets the session and restores the saved model
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path_unpruned.format('best'))

        # Reset the global step counter and begin pruning
        sess.run(reset_global_step_op)
        for epoch in range(epochs_prune):
            for batch in range(batches):
                batch_xs, batch_ys = sample_batch(X_train, y_train_binary, batch_size)
                # Prune and retrain
                sess.run(prune_op)
                sess.run(train_op, feed_dict={image: batch_xs, label: batch_ys})

            # Calculate Test Accuracy every epoch
            acc_print = 0
            acc_print_5 = 0
            for index, offset in enumerate(range(0, X_test.shape[0], batch_size)):
                batch_xt = np.array(X_test[offset: offset + batch_size,:])
                batch_yt = np.array(y_test_binary[offset: offset +  batch_size])
                acc_print += sess.run(accuracy, feed_dict={image: batch_xt, label: batch_yt})
                acc_print_5 += sess.run(accuracy_5, feed_dict={image: batch_xt, label: batch_yt})
            acc.append(acc_print/batches_test)
            acc_5.append(acc_print_5/batches_test)
            print(epoch)
            print("Pruned model step %d test accuracy %g" % (epoch, acc_print/batches_test))
            print("Pruned model step %d test top5-accuracy %g" % (epoch, acc_print_5/batches_test))
            print("Weight sparsities:", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))

        # Saves the model after pruning
        saver.save(sess, model_path_pruned.format(s))

        # Print final accuracy
        print("Final accuracy:", acc_print/batches_test)
        print("Final sparsity by layer", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))



## Plotting Training graph for 0.8 Sparsity model
iterations = [0,1,2,3,4]
training_acc = [0.8272, 0.8405, 0.8401, 0.8413, 0.8416]

plt.figure()
plt.plot(iterations,training_acc)
plt.title('Training after pruning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(Top 5)')
plt.show()



# Plotting the Trade-off between Accuracy and Sparsity
plt.figure()
plt.plot(sparsity, acc[0:55:5], 'b--.', label='after pruning')
plt.plot(sparsity, acc[4:55:5], 'k--.', label='pruning with training')
plt.axhline(y=acc_unpruned, color='r', linestyle='-', label = 'Un-pruned')
plt.annotate('{0:.4f}'.format(acc_unpruned), xy=(1, acc_unpruned), xytext=(8, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.title('Accuracy vs Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy(Top 1)')
plt.ylim(bottom=0, top=1)
plt.legend(loc = 3)
plt.show()

plt.figure()
plt.plot(sparsity, acc_5[0:55:5], 'b--x', label='after pruning')
plt.plot(sparsity, acc_5[4:55:5], 'k--x', label='pruning with training')
plt.axhline(y=acc_unpruned_5, color='r', linestyle='-', label = 'Un-Pruned')
plt.annotate('{0:.4f}'.format(acc_unpruned_5), xy=(1, acc_unpruned_5), xytext=(8, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.title('Accuracy(Top 5) vs Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy(Top 5)')
plt.ylim(bottom=0)
plt.legend(loc = 3)
plt.show()



thresh = []
pruned = []
not_pruned = []
weights_pruned = []
n_weights = []
with tf.Session() as sess:
    # Resets the session and restores the saved model
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, model_path_unpruned.format('best'))
    saver.restore(sess, model_path_pruned.format(0.8))
    # get the graph
    g = tf.get_default_graph()
    for i in range(7):
        if (i==0):
            t = g.get_tensor_by_name('Conv/threshold:0')
            m = g.get_tensor_by_name('Conv/mask:0')
            w = g.get_tensor_by_name('Conv/weights:0')

        else:
            t = g.get_tensor_by_name('Conv_{}/threshold:0'.format(i))
            m = g.get_tensor_by_name('Conv_{}/mask:0'.format(i))
            w = g.get_tensor_by_name('Conv_{}/weights:0'.format(i))

        x = sess.run(t)
        y = sess.run(m)
        z = sess.run(w)
        thresh.append(x)
        pruned.append(np.count_nonzero(y==0))
        not_pruned.append(np.sum(y))
        n_weights.append(np.size(y))
        weights_pruned.append(np.multiply(z,y))

print("Total Weights before Pruning: %.5fM" % (sum(n_weights) / 1e6))
print("Total Weights after Pruning: %.5fM" % (sum(not_pruned) / 1e6))
print("Total Weights pruned: %.5fM" % (sum(pruned) / 1e6))
print("Threshold for each layer:")
print(thresh)



# data to plot
n_layers = 7

# create plot
fig, ax = plt.subplots()
index = np.arange(n_layers)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, weights, bar_width,
alpha=opacity,
color='b',
label='Before Pruning')

rects2 = plt.bar(index + bar_width, not_pruned, bar_width,
alpha=opacity,
color='g',
label='After Pruning')

# Add counts above the two bar graphs
for i,rect in enumerate(rects1):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, 't = {0:.2f}'.format(thresh[i]),
             ha='center', va='bottom')

plt.xlabel('Convolutional Layers')
plt.ylabel('Number of parameters')
plt.title('Distribution of Weights in each Convolutional layer')
plt.xticks(index + bar_width/2, ('1', '2', '3', '4', '5', '6', '7'))
# Create empty plot with blank marker containing the extra legend
plt.plot([], [], ' ', label="t = Threshold")
plt.legend()

plt.tight_layout()
plt.show()



#Getting weights from Unpruned Model
weights_unpruned = []
with tf.Session() as sess:
    # Resets the session and restores the saved model
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path_unpruned.format('best'))
    tvars = tf.trainable_variables()
    for var in tvars:
        if var.name.endswith('weights:0'):
            val = sess.run(var)
            weights_unpruned.extend(val.flatten())
            #print(var.name, val)


# Removing zeros from Pruned weights and flattening
weights_pruned_2 = []
for wei in weights_pruned:
    wei = wei[wei!=0]
    weights_pruned_2.extend(wei)



plt.figure()
n, bins, patches = plt.hist(weights_unpruned, 100, range = (-0.5,0.5), alpha=0.6)
plt.xlabel('Magnitude of weights')
plt.ylabel('Number of weights')
plt.title('Distribution of the magnitude of Weights - Unpruned Model')
plt.show()




plt.figure()
n, bins, patches = plt.hist(weights_pruned_2, 100, range = (-0.5,0.5), alpha=0.6)
plt.xlabel('Magnitude of weights')
plt.ylabel('Number of weights')
plt.title('Distribution of the magnitude of Weights - Pruned Model')
plt.show()
