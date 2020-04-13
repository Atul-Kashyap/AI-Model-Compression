from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary
 #
 # model_dir ='Models/frozen_model.pb'
 # log_dir = 'Models/'
def import_to_tensorboard(model_dir, log_dir):
  with session.Session(graph=ops.Graph()) as sess:
    with gfile.FastGFile(model_dir, "rb") as f:
      graph_def = graph_pb2.GraphDef()
      graph_def.ParseFromString(f.read())
      importer.import_graph_def(graph_def)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))


def main(unused_args):
  import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      model_dir ='Models/frozen_model.pb',
      type=str,
      default="",
      required=True,
      help="The location of the protobuf (\'pb\') model to visualize.")
  parser.add_argument(
     log_dir='Models/',
      type=str,
      default="",
      required=True,
      help="The location for the Tensorboard log to begin visualization from.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
