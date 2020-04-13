from utils import *

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'notebook')


print(tf.__version__)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


model_dir = 'Models'
model_path_unpruned = os.path.join(model_dir,'Unpruned_{}.ckpt')
model_path_pruned = os.path.join(model_dir,'Pruned_{}.ckpt')
model_path_quantized = os.path.join(model_dir,'quantized_model.tflite')
model_path_pruned_quantized = os.path.join(model_dir,'quantized_pruned_model.tflite')
model_frozen = os.path.join(model_dir,'frozen_model.pb')
NUM_CLASS = 10


## Quantizing the tensorflow model to int8
tf.reset_default_graph()
image = tf.placeholder(name='images', dtype=tf.float32, shape=[None, 32, 32, 3])
label = tf.placeholder(name='fine_labels', dtype=tf.int32, shape=[None, 10])
logits = tf_fcn_model(image)

# Accuracy ops
#prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=tf.argmax(label, 1), k=5), tf.float32))

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path_pruned.format('0.8'))
    converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [image], [logits])
    tf.logging.set_verbosity(tf.logging.INFO)
    converter.post_training_quantize = True
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] #For tf 1.13.0
    tflite_quant_model = converter.convert()
    open(model_path_pruned_quantized, "wb").write(tflite_quant_model)

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path = model_path_pruned_quantized)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)



## To visualize and download the weights from tflite model
import netron
netron.start(model_path_pruned_quantized, port = 8000, host = 'localhost')



tflite_weights_path = 'Models/tflite'
weights = []

for file in os.listdir(tflite_weights_path):
    kernel = np.load(os.path.join(tflite_weights_path,file))
    print(kernel.shape)
    weights.extend(kernel.flatten())
print(len(weights))



plt.figure()
n, bins, patches = plt.hist(weights, 100, alpha=0.6)
plt.xlabel('Magnitude of weights')
plt.ylabel('Number of weights')
plt.title('Distribution of the Quantized Weights')
plt.show()



s_ckpt = os.stat(model_path_unpruned.format('best')+'.data-00000-of-00001').st_size/1e6
s_pb = os.stat(model_frozen).st_size/1e6
s_quant = os.stat(model_path_quantized).st_size/1e6

print("Size of Checkpoint file: {} Mbytes"
      .format(s_ckpt))
print("Size of Frozen model: {} Mbytes".format(s_pb))
print("Size of Quantized model: {} Mbytes".format(s_quant))

print("Quantized Model is {0:.2f} times smaller than Unquantized model".format(s_pb/s_quant))



objects = ('Checkpoint', 'Frozen Model(.pb)', 'Quantized')
y_pos = np.arange(len(objects))
performance = [s_ckpt, s_pb, s_quant]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Size in Mbytes')
plt.title('Model size Comparison')

plt.show()
