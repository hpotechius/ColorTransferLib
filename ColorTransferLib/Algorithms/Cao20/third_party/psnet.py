from .tf_ops import *
import os
import numpy as np
from pprint import pprint as ppt
from tensorflow.keras import layers  # neu


class PSNet():
    def __init__(self,config, sess, train_dir,
                 mode = "train",
                 num_pts = 4096,
                 late_fusion = True,
                 **st_dict):
        # define model structure
        self.fe_layer_units = list(map(lambda x: int(x), config["modelstructure"]["fe_layer"].split(",")))
        self.fc_layer_units = list(map(lambda x: int(x), config["modelstructure"]["fc_layer"].split(",")))

        self.num_label = config["training"].getint("num_label")
        self.base_lr = config["training"].getfloat("lr")
        self.lr_decay_rate = config["training"].getfloat("lr_decay_rate")
        self.lr_decay_step = config["training"].getfloat("lr_decay_step")
        self.mode = mode
        self.summary_loss = dict()
        self.summary_grad = dict()
        self.summary_hist = dict()
        self.node = dict()
        self.node_color = dict()
        self.node_geo = dict()
        self.kernal = dict()
        # layers used to extract content representation and style representation
        self.content_layer = list(map(lambda x: int(x),config["style_transfer"]["content_layer"].split(",")))
        self.style_layer = list(map(lambda x: int(x),config["style_transfer"]["content_layer"].split(",")))

        self.batch_size = config["training"].getint("batch_size")
        self.num_points = num_pts

        # input placeholders
        if mode == "styletransfer":
            self.from_image = st_dict["from_image"]
            if config["style_transfer"]["color_init"].startswith("r"):
                self.color = tf.compat.v1.get_variable(name="point_cloud_color",shape=[1, self.num_points, 3],
                                            initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.5))
            elif config["style_transfer"]["color_init"].startswith("c"):
                self.color = tf.compat.v1.get_variable(name="point_cloud_color", shape=[1, self.num_points, 3],
                                           initializer=st_dict["color_init"])
            self.summary_hist["color"] = tf.compat.v1.summary.histogram(name="color", values=self.color)
            if config["style_transfer"]["geo_init"].startswith("r") or self.from_image:
                self.geo = tf.compat.v1.get_variable(name="point_cloud",shape=[1, self.num_points, 3],
                                      initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.5))
            elif config["style_transfer"]["geo_init"].startswith("c"):
                self.geo = tf.compat.v1.get_variable(name="point_cloud", shape=[1, self.num_points, 3],
                                           initializer=st_dict["geo_init"])
            self.summary_hist["pts"] = tf.compat.v1.summary.histogram(name="pts", values=self.geo)
            self.beta_geo = config["style_transfer"].getint("beta_geo")
            self.pointclouds_pl = tf.concat([self.geo, self.color], axis=-1)

            self.target_content_representation = st_dict["target_content"]
            self.target_style_representation = st_dict["target_style"]
            self.st_optimizer_type = config["style_transfer"]["optimizer_type"].split("_")[0]
            self.st_lr = float(config["style_transfer"]["optimizer_type"].split("_")[1])
            self.beta_color = config["style_transfer"].getint("beta_color")

        elif mode == "train" or "test":
            self.geo = tf.compat.v1.placeholder(tf.float32,
                                         shape=[None, num_pts, 3], name="pc_geo")
            self.color = tf.compat.v1.placeholder(tf.float32,
                                           shape=[None, num_pts, 3], name="pc_color")
            #self.pointclouds_pl = tf.placeholder(tf.float32,
             #   shape=[None, self.num_points, 6], name="point_clouds")
            self.labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(None,), name = "label")
        self.bn_pl = tf.compat.v1.placeholder(dtype=tf.bool, shape=(), name="is_bn_training")
        self.dropout_prob_pl = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name="dropout_prob")
        self.late_fusion = late_fusion

        self.sess = sess
        # construct learning rate strategy
        self.batch_step = tf.Variable(0, name="batch_step")
        self._get_learning_rate()
        # build model

        self._build_model()
        # summary writer and moder save
        if mode == "train":
            self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir, sess.graph) # already saved graph
        elif self.mode.startswith("s"):
            #self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir, sess.graph)
            self.model_saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="cls"))
        if self.mode.startswith("t"):
            self.model_saver = tf.compat.v1.train.Saver(max_to_keep=200)
        # logfile
        #self.log_fout = open(os.path.join(train_dir, 'log.txt'), 'a')
        self.sess.run(tf.compat.v1.global_variables_initializer())


    def _fe_layers(self, input_features, scope):
        with tf.compat.v1.variable_scope(scope):
            net = input_features
            for idx, unit in enumerate(self.fe_layer_units):
                after_relu, net = FE_layer(net, unit,aggregate_global=True, bn_is_training=self.bn_pl,scope="FE_{}".format(idx + 1))
                self.node[scope + "_FE_" + str(idx + 1)] = after_relu
                if "COLOR" in scope:
                   self.node_color[scope + "_FE_" + str(idx + 1)] = after_relu
                if "PTS" in scope:
                    self.node_geo[scope + "_FE_" + str(idx + 1)] = after_relu
                self.kernal[scope + "_FE_" + str(idx + 1)] =  tf.compat.v1.get_default_graph().get_tensor_by_name("cls/" + scope + "/FE_{}/dense/kernel:0".format(idx + 1))
                # self.summary[scope + "_FE_" + str(idx + 1)] = tf.summary.histogram(scope + "_FE_" + str(idx + 1), self.kernal[scope + "_FE_" + str(idx + 1)])
            global_aggregated_feature = tf.reduce_mean(input_tensor=net, axis=1, name="aggregation")  # batch_size, 4096
        return global_aggregated_feature


    def _fc_layers(self, input_vectors, scope):
        with tf.compat.v1.variable_scope(scope):
            net = input_vectors
            for idx, unit in enumerate(self.fc_layer_units):
                net = dense_norm_nonlinear(net, unit, norm_type="bn",is_training=self.bn_pl, scope="FC_{}".format(idx + 1))
                self.node[scope + "_FC_" + str(idx + 1)] = net
                net = tf.nn.dropout(net, rate=1 - (self.dropout_prob_pl), name="dropout_{}".format(idx + 1))
            # logits = tf.compat.v1.layers.dense(net, self.num_label)
            logits = layers.Dense(
                units=self.num_label,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                name="dense",
            )(net)
        return logits


    def _build_model(self):
        # graph
        with tf.compat.v1.variable_scope("cls"):
            if self.late_fusion:
                # if self.mode.startswith("t"):
                #     self.geo = self.pointclouds_pl[...,:3]
                #     self.color = self.pointclouds_pl[...,3:6]
                self.color_aggregated = self._fe_layers(self.color, "FE_COLOR")  # batchsize, num_units
                self.pts_aggregated = self._fe_layers(self.geo, "FE_PTS")  # batchsize, num_units
                self.pts_color_aggregated = tf.concat([self.pts_aggregated, self.color_aggregated], -1)
            else:
                self.pts_color_aggregated = self._fe_layers(self.pointclouds_pl, "FE_PTS_COLOR")
            self.logits = self._fc_layers(self.pts_color_aggregated, scope="FC")

        # loss
        if self.mode.startswith("t"):
            print("Building cross entropy loss...")
            self.loss = tf.reduce_mean(
                input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_pl), name="cross_entropy")
            loss_sum = tf.compat.v1.summary.scalar(name="training_loss", tensor=self.loss)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr) \
                    .minimize(self.loss, global_step=self.batch_step)
        elif self.mode.startswith("s"):
            print("Optimizer: {0} Learning Rate: {1}".format(self.st_optimizer_type, self.st_lr))
            # style transfer loss
            if self.st_optimizer_type == "sgd":
                self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.st_lr)
            elif self.st_optimizer_type == "adadelta":
                self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.st_lr, rho=0.9)
            elif self.st_optimizer_type == "adagrad":
                self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.st_lr)
            elif self.st_optimizer_type == "adam":
                self.optimizer = tf.compat.v1.train.AdamOptimizer(self.st_lr)
            elif self.st_optimizer_type == "momentum":
                self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.st_lr, momentum=0.9, use_nesterov= False)
            elif self.st_optimizer_type == "rmsprop":
                self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.st_lr)
            elif self.st_optimizer_type == "nesterov":
                self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.st_lr, momentum=0.9, use_nesterov= True)
            else:
                raise ValueError("Please choose the correct optimizer for style transfer!")

            print("Building style transfer loss...")
            if self.late_fusion:
                if not self.from_image:
                    use_content_geo = ["FE_PTS_FE_{}".format(i) for i in self.content_layer]
                    use_style_geo = ["FE_PTS_FE_{}".format(i) for i in self.style_layer]
                    # content loss for geo
                    loss_geo_content = []
                    for layer in use_content_geo:
                        loss_geo_content.append(
                            tf.nn.l2_loss(self.target_content_representation[layer] - self.node[layer]) / tf.size(
                                input=self.node[layer], out_type=tf.float32))
                    self.loss_geo_content = tf.add_n(loss_geo_content)

                    # style loss for geo
                    loss_geo_gram = []
                    for layer in use_style_geo:
                        source_gram_pts = tf.matmul(tf.transpose(a=tf.squeeze(self.node[layer])),
                                                    tf.squeeze(self.node[layer])) / tf.size(input=self.node[layer],
                                                                                            out_type=tf.float32)
                        loss_geo_gram.append(tf.nn.l2_loss(self.target_style_representation[layer] - source_gram_pts))
                    self.loss_geo_style = tf.add_n(loss_geo_gram)
                    self.total_loss_geo = self.loss_geo_content + self.beta_geo * self.loss_geo_style
                    self.summary_loss["loss_geo"] = tf.compat.v1.summary.scalar("loss_geo", self.total_loss_geo)
                    self.summary_loss["loss_geo_content"] = tf.compat.v1.summary.scalar("loss_geo_content", self.loss_geo_content)
                    self.summary_loss["loss_geo_style"] = tf.compat.v1.summary.scalar("loss_geo_style", self.loss_geo_style)

                use_content_color = ["FE_COLOR_FE_{}".format(i) for i in self.content_layer]
                use_style_color = ["FE_COLOR_FE_{}".format(i) for i in self.style_layer]
                # content loss for color
                loss_color_content = []
                for layer in use_content_color:
                    loss_color_content.append(
                        tf.nn.l2_loss(self.target_content_representation[layer] - self.node[layer]) / tf.size(input=self.node[layer],out_type=tf.float32))
                self.loss_color_content  = tf.add_n(loss_color_content)

                # style loss for color
                loss_color_gram = []
                for layer in use_style_color:
                    source_gram_color = tf.matmul(tf.transpose(a=tf.squeeze(self.node[layer])),
                                                  tf.squeeze(self.node[layer])) / tf.size(input=self.node[layer],
                                                                                          out_type=tf.float32)
                    loss_color_gram.append(tf.nn.l2_loss(self.target_style_representation[layer] - source_gram_color))
                self.loss_color_style = tf.add_n(loss_color_gram)

                # total loss
                self.total_loss_color = self.loss_color_content + self.beta_color * self.loss_color_style
                self.summary_loss["loss_color"] = tf.compat.v1.summary.scalar("loss_color", self.total_loss_color)
                self.summary_loss["loss_color_content"] = tf.compat.v1.summary.scalar("loss_color_content", self.loss_color_content)
                self.summary_loss["loss_color_style"] = tf.compat.v1.summary.scalar("loss_color_style", self.loss_color_style)

                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optim_op_color = self.optimizer \
                        .minimize(self.total_loss_color, global_step=self.batch_step, var_list=[self.color])
                    if not self.from_image:
                        self.optim_op_pts = self.optimizer \
                            .minimize(self.total_loss_geo, global_step=self.batch_step, var_list=[self.geo])
            else: # early fusion
                use_content = ["FE_PTS_COLOR_FE_{}".format(i) for i in self.content_layer]
                use_style = ["FE_PTS_COLOR_FE_{}".format(i) for i in self.style_layer]
                print("Building style transfer loss...")
                # content loss
                loss_content = []
                for layer in use_content:
                    loss_content.append(
                        tf.nn.l2_loss(self.target_content_representation[layer] - self.node[layer]) / tf.size(
                            input=self.node[layer], out_type=tf.float32))
                self.loss_content = tf.add_n(loss_content)

                # style loss
                loss_style_gram = []
                for layer in use_style:
                    source_gram = tf.matmul(tf.transpose(a=tf.squeeze(self.node[layer])),
                                                  tf.squeeze(self.node[layer])) / tf.size(input=self.node[layer],
                                                                                          out_type=tf.float32)
                    loss_style_gram.append(tf.nn.l2_loss(self.target_style_representation[layer] - source_gram))
                self.loss_style = tf.add_n(loss_style_gram)
                # total loss
                self.total_loss = self.loss_content + self.beta_geo * self.loss_style
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optim_op_color = self.optimizer \
                        .minimize(self.total_loss, global_step=self.batch_step, var_list=[self.color])
                    self.optim_op_pts = self.optimizer \
                        .minimize(self.total_loss, global_step=self.batch_step, var_list=[self.geo])
                    self.optim_op_total = self.optimizer \
                        .minimize(self.total_loss, global_step=self.batch_step, var_list=[self.geo, self.color])

        else:
            raise ValueError("please choose the right mode!current mode is {}".format(self.mode))

        # summary
        # self.summary["grad_weight_FE1_pts"] = tf.summary.histogram("grad_weight_FE1_pts", tf.get_default_graph().\
        #     get_tensor_by_name("gradients/FE_PTS/FE_1/dense/Tensordot/MatMul_grad/MatMul_1:0"))
        # self.summary["grad_weight_FE1_color"] = tf.summary.histogram("grad_weight_FE1_color", tf.get_default_graph(). \
        #     get_tensor_by_name("gradients/FE_COLOR/FE_1/dense/Tensordot/MatMul_grad/MatMul_1:0"))

        # grad_color = tf.get_default_graph().get_tensor_by_name(
        #     "gradients/cls/FE_COLOR/FE_1/dense/Tensordot/MatMul_grad/MatMul_1:0")
        # self.summary["grad_color"] = tf.summary.histogram("grad_color", grad_color)
        # grad_pts = tf.get_default_graph().get_tensor_by_name(
        #     "gradients/cls/FE_PTS/FE_1/dense/Tensordot/MatMul_grad/MatMul_1:0")
        # self.summary["grad_pts"] = tf.summary.histogram("grad_pts", grad_pts)
        self.all_summary = tf.compat.v1.summary.merge_all()

        return True


    def style_transfer_one_step(self, update_property = "geometry"):
        if self.late_fusion:
            train_dict_color = self.sess.run({"loss_color": self.total_loss_color,
                                              "loss_color_content": self.loss_color_content,
                                              "loss_color_style": self.loss_color_style,
                                              "optim_op": self.optim_op_color,
                                              "batch_step": self.batch_step},
                                             feed_dict={
                                                 self.bn_pl: False,
                                                 self.dropout_prob_pl: 1.0})
            self.log_string("step: {0:5d} st_color_loss: {1:.8f} color_loss_content: {2:.8f} color_loss_style: {3:.8f}".
                            format(train_dict_color["batch_step"], train_dict_color["loss_color"],
                                   train_dict_color["loss_color_content"], train_dict_color["loss_color_style"]))
            if not self.from_image:
                train_dict_pts = self.sess.run({"loss_geo": self.total_loss_geo,
                                                "loss_geo_content":self.loss_geo_content,
                                                "loss_geo_style": self.loss_geo_style,
                                               "optim_op": self.optim_op_pts,
                                               "batch_step": self.batch_step},
                                              feed_dict={
                                                  self.bn_pl: False,
                                                  self.dropout_prob_pl:1.0})
                self.log_string("step: {0:5d} st_pts_loss: {1:.8f} pts_loss_content: {2:.8f} pts_loss_style: {3:.8f}".
                                format(train_dict_pts["batch_step"], train_dict_pts["loss_geo"],
                                       train_dict_pts["loss_geo_content"], train_dict_pts["loss_geo_style"]))
            # geo only
            # self.log_string("step: {0:5d} st_color_loss: {1:.3f} "
            #                 "\n pts_loss_content: {2:.3f} pts_loss_style:{3:.3f}".
            #                 format(train_dict_pts["batch_step"], train_dict_pts["loss_pts"],
            #                        train_dict_pts["loss_pts_content"], train_dict_pts["loss_pts_style"]))
            # # color only
            # self.log_string("step: {0:5d} st_color_loss: {1:.3f} "
            #                 "\n color_loss_content: {2:.3f} color_loss_style:{3:.3f}".
            #                 format(train_dict_color["batch_step"], train_dict_color["loss_color"],
            #                        train_dict_color["loss_color_content"], train_dict_color["loss_color_style"]))
        else: # early fusion
            if update_property.startswith("g"):
                op = self.optim_op_pts
            elif update_property.startswith("c"):
                op = self.optim_op_color
            elif update_property.startswith("t"):
                op = self.optim_op_total
            train_dict_pts = self.sess.run({"loss_total": self.total_loss,
                                            "loss_content": self.loss_content,
                                            "loss_style": self.loss_style,
                                            "optim_op": op,
                                            "batch_step": self.batch_step},
                                           feed_dict={
                                               self.bn_pl: False,
                                               self.dropout_prob_pl: 1.0})
            self.log_string("step: {0:5d} loss_total: {1:.5f} loss_content: {2:.5f} loss_style: {3:.5f}".format(
                                   train_dict_pts["batch_step"],
                                   train_dict_pts["loss_total"],
                                   train_dict_pts["loss_content"],
                                   train_dict_pts["loss_style"]))




    def _get_learning_rate(self):
        learning_rate = tf.compat.v1.train.exponential_decay(
            self.base_lr,  # Base learning rate.
            self.batch_step * self.batch_size,  # Current index into the dataset.
            self.lr_decay_step,  # Decay step.
            self.lr_decay_rate,  # Decay rate.
            staircase=True)
        self.lr = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        self.sum_lr = tf.compat.v1.summary.scalar(name="learning_rate", tensor=self.lr)
        return True


    def save_model(self, path, i):
        """
        save model parameters
        :param path:
        :return:
        """
        self.model_saver.save(self.sess, path, i)
        return True


    def restore_model(self, model_path):
        """
        restore model graph and parameters from path
        :param model_path:
        :return:
        """
        self.model_saver.restore(self.sess, model_path)
        return True


    def train_one_batch(self, batch_data, batch_label):
        all_sum, _, loss, batch_step = self.sess.run([self.all_summary, self.optim_op, self.loss, self.batch_step],
                              feed_dict={
                                  self.pointclouds_pl: batch_data,
                                  self.labels_pl: batch_label,
                                  self.bn_pl: True,
                                  self.dropout_prob_pl:0.7})
        self.summary_writer.add_summary(all_sum, batch_step)
        self.log_string(
            "global_step:{0:5d} train_loss: {1}".format(batch_step, loss))
        return True


    def eval_one_batch(self, batch_data, batch_label):
        eval_dict = self.sess.run({"eval_logtis": self.logits}, feed_dict={
                                  self.pointclouds_pl: batch_data,
                                  self.labels_pl: batch_label,
                                  self.bn_pl: False,
                                  self.dropout_prob_pl:1.0})
        return eval_dict

    # def eval_one_epoch(self,):


    def log_string(self, out_str):
        #self.log_fout.write(out_str + '\n')
        #self.log_fout.flush()
        print(out_str)
