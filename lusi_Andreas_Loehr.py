from cgi import test
from pyexpat import model
from re import L
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import functools
from  pprint import pprint
from scipy import ndimage as ndimage
import lusi_periphery


class LusiModel(tf.keras.Model):
    """Given a model, e.g. neural net, implement the LUSI approach.
    
    Note: This code has only been tested with neural nets defined via
          keras API.
    """
    
    def __init__(self, m_inner_prod, model=None, tau=None) -> None:
        """Instatiate model with custom training loop adapted to LUSI method.

        Parameters:

        m_inner_prod :: tf.Variable with appropriate dims, format tf.float32.
            Matrix used for custom inner product in Lusi Loss ('M' in paper).

        model :: tensorflow model | None.
            Underlying model to be used to make predictions.
            If no model is passed, use simple neural net as default.

        tau :: np.ndarray of functions | None.
            Predicates of the form f : \mathcal{Y} -> \R.
            See paper section 'factorizing predicates'.
        """
        
        super().__init__()  # This may not be necessary and cause problems.
        if not model:
            # if no model given, use this small neural net as default.
            self.model = keras.Sequential(
                [layers.Flatten(input_shape=(28,28)),
                 layers.Dense(100, activation="relu", name="hidden_layer_01"),
                 layers.Dense(1, name="output_layer", activation="sigmoid") # interpret output as prob. for class 1
                ]
                )
        else:
            self.model = model

        self.m_inner_prod = m_inner_prod
        self.tau = tau

        return None


    def summary(self):
        """Display model summary."""

        return self.model.summary()
    
    def add_optimizer(self, optimizer) -> None:
        """Add tf optimizer to use."""

        self.optimizer = optimizer

        return None


    def train(self, train_dataset, num_epochs, train_metrics=[], verbose=True):
        """Training loop for custom training with LUSI loss.
    
        Parameters:

        train_dataset :: zipped BatchDataSet
            train_dataset consists of 2 zipped datasets. Each of the 2
            datasets in the zip object must consist of the following triple
                (phi_evaluated, x, y).
            Use the Periphery.generate_batch_data method from
            lusi_periphery.py to obtain correct structure.
        
        num_epochs :: int
            Number of epochs to train.
        
        train_metrics :: list[metrics]
            A list of metrics to evaluate progress on. These metrics should
            be tf.keras.metrics instances which have been processed by the
            modify_metric function defined in this script.
        """

        if not self.optimizer:
            raise TypeError("No optimizer specified. \
                            Please use add_optimizer to add optimizer.")
        
        if verbose:
            gradient_list = []
            epoch_train_metrics_results = []
            model_weight_list = []
            model_weight_list.append((-1, self.layers[0].get_weights()))
        
        # sanity check matrix dims and predicates match
        if not train_dataset.element_spec[0][0].shape[1] == \
            self.m_inner_prod.shape[0]:
                raise ValueError("Check if dims of m_inner_prod and no. of \
                                 predicates match.")
        
        for epoch in range(num_epochs):
            if verbose:
                print("\nStart of epoch %d" % (epoch,))   


            # Iterate over the batches of the dataset.
            for step, ((pred_batch_1, x_batch_1, y_batch_1), 
                (pred_batch_2, x_batch_2, y_batch_2))  \
                    in enumerate(train_dataset):
                
                
                # need total batch for evaluation and calc of 'true' loss.
                x_total = tf.concat([x_batch_1, x_batch_2], axis=0)
                pred_total = tf.concat([pred_batch_1, pred_batch_2], axis=0)
                
                # y tensors must be of dim nx1
                y_batch_1 = tf.cast(y_batch_1, tf.float32)
                y_batch_1= tf.expand_dims(y_batch_1, axis=1)
                y_batch_2 = tf.cast(y_batch_2, tf.float32)
                y_batch_2= tf.expand_dims(y_batch_2, axis=1)
                y_total = tf.concat([y_batch_1, y_batch_2], axis=0)
                
                # Predictions on 2nd batch. This calc should not be recorded
                # on gradient tape s.t. values are treated as constants during
                # automatic differentiation.
                y_batch_2_pred = self.model(x_batch_2)

                # broadcasting difference with actual labels to shape which
                # is compatible for Hadamard product with predicates.
                # result: d identical columns where d = no. of predicates.
                v_prime_inter = tf.broadcast_to(y_batch_2_pred - y_batch_2,
                                                shape=[y_batch_2.shape[0],
                                                pred_batch_2.shape[1]])

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Logits for batch J recorded on gradient tape 
                    y_batch_1_pred = self.model(x_batch_1)

                    # prepare for multipication with predicate evaluations
                    # result: d identical columns where d = no. of predicates.
                    y_batch_1_pred_broad = tf.broadcast_to(y_batch_1_pred,
                        shape=[y_batch_1_pred.shape[0],
                               pred_batch_1.shape[1]])
                    
                    # Compute the loss value to be differentiated for batch.
                    v = tf.reduce_mean(pred_batch_1 * y_batch_1_pred_broad,
                                       axis=0, keepdims=True)

                    # v_prime_inter = tf.broadcast_to(y_batch_2_pred - y_batch_2, 
                    #                                 shape=[y_batch_2.shape[0],
                    #                                 pred_batch_2.shape[1]])

                    v_prime = tf.reduce_mean(pred_batch_2 * v_prime_inter,
                                            axis=0, keepdims=True)
                    
                    # v_prime_times_weight_matrix = tf.matmul(self.m_inner_prod,
                    #                                   tf.transpose(v_prime))

                    loss_value = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                    tf.tensordot(v,
                                        tf.matmul(self.m_inner_prod,
                                            tf.transpose(v_prime)), axes=1))

                
                y_pred_total = tf.concat([y_batch_1_pred, y_batch_2_pred],
                                         axis=0)
                
                y_pred_total_inter = tf.broadcast_to(y_total - y_pred_total,
                                        shape=[y_pred_total.shape[0],
                                            pred_total.shape[1]])

                v_actual_loss = tf.reduce_mean(pred_total * \
                                               y_pred_total_inter,
                                               axis=0, keepdims=True)

                actual_loss = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                  tf.tensordot(v_actual_loss,
                                      tf.matmul(self.m_inner_prod,
                                          tf.transpose(v_actual_loss)),
                                          axes=1))
                if verbose:
                    watched_vars = tape.watched_variables()
                
                    for train_metric in train_metrics:
                        if train_metric.expected_input == "pred_and_true":
                            train_metric.update_state(y_total, 
                                tf.round(self.model(x_total)))

                        elif train_metric.expected_input == "loss":
                            train_metric.update_state(actual_loss)
                
                    # debugging only - add metric scores before first update
                    if epoch == 0 and step==0:
                        epoch_train_metrics_results.append(
                        [(train_metric.name, train_metric.result().numpy()) \
                            for train_metric in train_metrics])  
                
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to 
                # the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                if verbose:
                    # storing gradients for further inspection
                    gradient_list.append(((epoch, step), grads))
                    
                    # storing model weights after each update
                    model_weight_list.append(((epoch, step), self.layers[0].get_weights()))

            if verbose:
                # Epoch stats
                epoch_train_metrics_results.append(
                    [(eval_metric.name, eval_metric.result().numpy()) \
                        for eval_metric in train_metrics])

                # reset metrics
                for eval_metric in train_metrics:
                    eval_metric.reset_state()
            
        if verbose: 
            self.epoch_train_metrics_results = epoch_train_metrics_results
            self.gradient_list = gradient_list
            self.model_weight_list = model_weight_list
            self.watches_vars = watched_vars
            
        return None


    def predict(self, x):
        """Predict outputs given inputs x."""

        return self.model(x)

    
    def evaluate(self, test_dataset, eval_metrics, training=False):
        """Evaluate model on test dataset given eval_metrics.
        
        Parameters:

        test_dataset :: tf batched dataset or tuple of np.ndarrays
        eval_metrics :: list of tf metrics proccessed by 'modify_metric'.        
        
        Returns:
        List of results from evaluation metrics calculated over test_dataset.
        """
        
        if not ((isinstance(test_dataset, tuple) and isinstance(
            test_dataset[0], np.ndarray)) or isinstance(test_dataset,
            tf.data.Dataset)):
            raise TypeError("Pass tf. Dataset or tuple of np.ndarrays")
        
        for eval_metric in eval_metrics:
                eval_metric.reset_state()
        
        if isinstance(test_dataset, tf.data.Dataset):

            for _, (pred_batch_test, x_batch_test, y_batch_test) in enumerate(test_dataset):
            
                for eval_metric in eval_metrics:
                    eval_metric.update_state(y_batch_test, tf.round(self.model(x_batch_test)))
        
            return [(eval_metric.name, eval_metric.result()) for eval_metric in eval_metrics]

        # tuple of numpy.ndarrays
        y_pred = self.predict(test_dataset[0])

        for eval_metric in eval_metrics:
            eval_metric.update_state(test_dataset[1], tf.round(y_pred))
                
        return [(eval_metric.name, eval_metric.result()) for eval_metric in eval_metrics]
        

class LusiErm(tf.keras.Model):
    """Given a model, e.g. neural net, implement the LUSI-ERM approach.
    
    Note: This code has only been tested with neural nets defined via
          keras API.
    """
    
    def __init__(self, m_inner_prod, alpha=0.5, model=None, erm_loss= None,
                 tau=None) -> None:
        """Instatiate model with custom training loop adapted to LUSI method.

        Parameters:

        m_inner_prod :: tf.Variable with appropriate dims, format tf.float32.
            Matrix used for custom inner product in Lusi Loss ('M' in paper).

        model :: tensorflow model | None.
            Underlying model to be used to make predictions.
            If no model is passed, use simple neural net as default.

        tau :: np.ndarray of functions | None.
            Predicates of the form f : \mathcal{Y} -> \R.
            See paper section 'factorizing predicates'.
        """
        
        super().__init__()  # This may not be necessary and cause problems.
        
        if not model:
            # if no model given, use this small neural net as default.
            self.model = keras.Sequential(
                [layers.Flatten(input_shape=(28,28)),
                 layers.Dense(100, activation="relu", name="hidden_layer_01"),
                 layers.Dense(1, name="output_layer", activation="sigmoid") # interpret output as prob. for class 1
                ]
            )
        else:
            self.model = model
        
        if not erm_loss:
            self.erm_loss = tf.keras.losses.BinaryCrossentropy()
        else:
            self.erm_loss = erm_loss

        self.m_inner_prod = m_inner_prod
        self.tau = tau
        self.alpha = alpha
        
        return None


    def summary(self):
        """Display model summary."""

        return self.model.summary()
    
    def add_optimizer(self, optimizer) -> None:
        """Add tf optimizer to use."""

        self.optimizer = optimizer

        return None
    
    def add_loss(self, loss) -> None:
        """Add an erm-loss function/object."""
        
        self.erm_loss = loss
        
        return None


    def train(self, train_dataset, num_epochs, train_metrics=[], verbose=True):
        """Training loop for custom training with LUSI loss.
    
        Parameters:

        train_dataset :: zipped BatchDataSet
            train_dataset consists of 2 zipped datasets. Each of the 2
            datasets in the zip object must consist of the following triple
                (phi_evaluated, x, y).
            Use the Periphery.generate_batch_data method from
            lusi_periphery.py to obtain correct structure.
        
        num_epochs :: int
            Number of epochs to train.
        
        train_metrics :: list[metrics]
            A list of metrics to evaluate progress on. These metrics should
            be tf.keras.metrics instances which have been processed by the
            modify_metric function defined in this script.
        """

        if not self.optimizer:
            raise TypeError("No optimizer specified. \
                            Please use add_optimizer to add optimizer.")
        
        if verbose:
            gradient_list = []
            epoch_train_metrics_results = []
            model_weight_list = []
            model_weight_list.append((-1, self.layers[0].get_weights()))
        
        # sanity check matrix dims and predicates match
        if not train_dataset.element_spec[0][0].shape[1] == \
            self.m_inner_prod.shape[0]:
                raise ValueError("Check if dims of m_inner_prod and no. of \
                                 predicates match.")
        
        for epoch in range(num_epochs):
            if verbose:
                print("\nStart of epoch %d" % (epoch,))   


            # Iterate over the batches of the dataset.
            for step, ((pred_batch_1, x_batch_1, y_batch_1), 
                (pred_batch_2, x_batch_2, y_batch_2))  \
                    in enumerate(train_dataset):
                
                
                # need total batch for evaluation and calc of 'true' loss.
                x_total = tf.concat([x_batch_1, x_batch_2], axis=0)
                pred_total = tf.concat([pred_batch_1, pred_batch_2], axis=0)
                
                # y tensors must be of dim nx1
                y_batch_1 = tf.cast(y_batch_1, tf.float32)
                y_batch_1= tf.expand_dims(y_batch_1, axis=1)
                y_batch_2 = tf.cast(y_batch_2, tf.float32)
                y_batch_2= tf.expand_dims(y_batch_2, axis=1)
                y_total = tf.concat([y_batch_1, y_batch_2], axis=0)
                
                # Predictions on 2nd batch. This calc should not be recorded
                # on gradient tape s.t. values are treated as constants during
                # automatic differentiation.
                y_batch_2_pred = self.model(x_batch_2)

                # broadcasting difference with actual labels to shape which
                # is compatible for Hadamard product with predicates.
                # result: d identical columns where d = no. of predicates.
                v_prime_inter = tf.broadcast_to(y_batch_2_pred - y_batch_2,
                                                shape=[y_batch_2.shape[0],
                                                pred_batch_2.shape[1]])

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Logits for batch J recorded on gradient tape 
                    y_batch_1_pred = self.model(x_batch_1)

                    # prepare for multipication with predicate evaluations
                    # result: d identical columns where d = no. of predicates.
                    y_batch_1_pred_broad = tf.broadcast_to(y_batch_1_pred,
                        shape=[y_batch_1_pred.shape[0],
                               pred_batch_1.shape[1]])
                    
                    # Compute the loss value to be differentiated for batch.
                    v = tf.reduce_mean(pred_batch_1 * y_batch_1_pred_broad,
                                       axis=0, keepdims=True)

                    # v_prime_inter = tf.broadcast_to(y_batch_2_pred - y_batch_2, 
                    #                                 shape=[y_batch_2.shape[0],
                    #                                 pred_batch_2.shape[1]])

                    v_prime = tf.reduce_mean(pred_batch_2 * v_prime_inter,
                                            axis=0, keepdims=True)
                    
                    # v_prime_times_weight_matrix = tf.matmul(self.m_inner_prod,
                    #                                   tf.transpose(v_prime))

                    loss_value = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                    tf.tensordot(v,
                                        tf.matmul(self.m_inner_prod,
                                            tf.transpose(v_prime)), axes=1))

                    # need to calc again
                    # y_pred_total = tf.concat([y_batch_1_pred, y_batch_2_pred],
                    #                         axis=0)
                    y_pred_total = self.model(x_total)

                    erm_plus_lusi = self.alpha * self.erm_loss(y_total, y_pred_total) + \
                        (1 - self.alpha) * loss_value
                

                y_pred_total_inter = tf.broadcast_to(y_total - y_pred_total,
                                        shape=[y_pred_total.shape[0],
                                            pred_total.shape[1]])

                v_actual_loss = tf.reduce_mean(pred_total * \
                                               y_pred_total_inter,
                                               axis=0, keepdims=True)

                actual_loss = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                  tf.tensordot(v_actual_loss,
                                      tf.matmul(self.m_inner_prod,
                                          tf.transpose(v_actual_loss)),
                                          axes=1))
                if verbose:
                    watched_vars = tape.watched_variables()
                
                    for train_metric in train_metrics:
                        if train_metric.expected_input == "pred_and_true":
                            train_metric.update_state(y_total, 
                                tf.round(self.model(x_total)))

                        elif train_metric.expected_input == "loss":
                            train_metric.update_state(actual_loss)
                
                    # debugging only - add metric scores before first update
                    if epoch == 0 and step==0:
                        epoch_train_metrics_results.append(
                        [(train_metric.name, train_metric.result().numpy()) \
                            for train_metric in train_metrics])  
                
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to 
                # the loss.
                grads = tape.gradient(erm_plus_lusi, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                if verbose:
                    # storing gradients for further inspection
                    gradient_list.append(((epoch, step), grads))
                    
                    # storing model weights after each update
                    model_weight_list.append(((epoch, step), self.layers[0].get_weights()))

            if verbose:
                # Epoch stats
                epoch_train_metrics_results.append(
                    [(eval_metric.name, eval_metric.result().numpy()) \
                        for eval_metric in train_metrics])

                # reset metrics
                for eval_metric in train_metrics:
                    eval_metric.reset_state()
            
        if verbose: 
            self.epoch_train_metrics_results = epoch_train_metrics_results
            self.gradient_list = gradient_list
            self.model_weight_list = model_weight_list
            self.watches_vars = watched_vars
            
        return None


    def predict(self, x):
        """Predict outputs given inputs x."""

        return self.model(x)

    
    def evaluate(self, test_dataset, eval_metrics, training=False):
        """Evaluate model on test dataset given eval_metrics.
        
        Parameters:

        test_dataset :: tf batched dataset or tuple of np.ndarrays
        eval_metrics :: list of tf metrics proccessed by 'modify_metric'.        
        
        Returns:
        List of results from evaluation metrics calculated over test_dataset.
        """
        
        if not ((isinstance(test_dataset, tuple) and isinstance(
            test_dataset[0], np.ndarray)) or isinstance(test_dataset,
            tf.data.Dataset)):
            raise TypeError("Pass tf. Dataset or tuple of np.ndarrays")
        
        for eval_metric in eval_metrics:
                eval_metric.reset_state()
        
        if isinstance(test_dataset, tf.data.Dataset):

            for _, (pred_batch_test, x_batch_test, y_batch_test) in enumerate(test_dataset):
            
                for eval_metric in eval_metrics:
                    eval_metric.update_state(y_batch_test, tf.round(self.model(x_batch_test)))
        
            return [(eval_metric.name, eval_metric.result()) for eval_metric in eval_metrics]

        # tuple of numpy.ndarrays
        y_pred = self.predict(test_dataset[0])

        for eval_metric in eval_metrics:
            eval_metric.update_state(test_dataset[1], tf.round(y_pred))
                
        return [(eval_metric.name, eval_metric.result()) for eval_metric in eval_metrics]


class ERM():
    "Wrap simple keras model for easier use with run_config function."
    
    def __init__(self, model=None) -> None:
        if not model:            
            # if no model given, use this small neural net as default.
            self.model = keras.Sequential(
                [layers.Flatten(input_shape=(28,28)),
                 layers.Dense(100, activation="relu", name="hidden_layer_01"),
                 layers.Dense(1, name="output_layer", activation="sigmoid") # interpret output as prob. for class 1
                ]
                )    
        else:
            self.model = model

        return None
        

    def compile(self, optimizer=keras.optimizers.SGD(),
                loss=keras.losses.BinaryCrossentropy(), metrics=[]):
        
        self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
                )
        return None
    
    def train(self, x_train, y_train, batch_size, epochs):
        """Call fit method of model."""

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        
        return None
    
    def evaluate(self, x_test, y_test, batch_size):
        """Call evaluate method of model."""
        self.model.evaluate(x_test, y_test, batch_size=batch_size)
        

        return None
        
        

def symmetry(imgs, axis="both"):
    """Calculate a symmetry score for normalized pictures.
    
    Parameters:
    img :: np.array
        Numpy array representing images. Pixel values in range [0,1]
    axis :: string, one of "vertical", "horizontal, "both"
        String indicating which axes to consider when calculating symmetries.


    Returns:
    sym_score :: float 
        Value representing symmetry - score between 0 (not symmetric) and 1 (symmetric).
    """
    
    # crop image to area of non-zero pixels only
    
    if len(imgs.shape) < 3:
        # only single image passed -> expand dims
        imgs = np.expand_dims(imgs, axis=0)
    
    if len(imgs.shape) < 4:
        # no channel for images -> add channel dim
        imgs = np.expand_dims(imgs, axis=3)
    
    if axis == "vertical":
        # print("vertical")
        flipped_imgs = tf.image.flip_left_right(imgs)

    elif axis == "horizontal":
        # print("horizontal")
        flipped_imgs = tf.image.flip_up_down(imgs)
    
    else:
        # diagonal
        # print("both v. and h.")
        flipped_imgs = tf.image.flip_up_down(tf.image.flip_left_right(imgs))
    
    # print(tf.reduce_mean(tf.abs(flipped_imgs - imgs), axis=[1, 2, 3]).shape)
    sym_score = tf.reduce_mean(1 - tf.reduce_mean(tf.abs(flipped_imgs - imgs), axis=[1, 2, 3]))
    
    return sym_score
    

def determine_box(imgs):
    """Determine non-black area of image.
    
    Provide coordinates for cropping images to area which is filled.
    
    Parameters: 
    
    img :: np.ndarray
    """
    
    if not len(imgs.shape) == 3:
        raise Exception("Check dimensions.")
    
    cropped_imgs = []
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :]
        hc = np.where(img!=0)[0]
        wc = np.where(img!=0)[1]
        ul = (np.min(hc),np.min(wc))
        lr = (np.max(hc), np.max(wc))
        cropped_imgs.append(np.expand_dims(img[ul[0]:lr[0], ul[1]:lr[1]], axis=(0,3)))
    return cropped_imgs


def avg_pixel_intensity(img_tensor, batch_mean=False):
    """Calculate avg. pixel intensity over image.
    
    Parameters:
    img_tensor :: array of 3 dim, pixel value between 0 and 1.
        Dimensions = (batch_size, width, height)
    """

    if batch_mean: 
        return tf.reduce_mean(tf.reduce_mean(img_tensor, axis=[1,2]), axis=0)
        
    # return tf.reduce_mean(img_tensor, axis=[1,2])
    return tf.reduce_mean(img_tensor, axis=[0,1])


def symmetry_boxed(imgs, axis="both", single=False):
    """Determine symmetry of cropped images.
    
    Assume [0,1]-range for pixels.
    """

    if single:
        # need (1, rows, cols) arrays for aply_phi function of Periphery class
        imgs = np.expand_dims(imgs, axis=0)

    cropped_imgs = determine_box(imgs)
    sym_scores = np.zeros(len(cropped_imgs))
    
    for j in range(len(imgs)):
        sym_scores[j] = symmetry(cropped_imgs[j], axis)
    
    return sym_scores


def weighted_pixel_intesity(x):
    """Determine weighted pixel intesity of image.
    
    Parameters:

    x :: np.ndarray
        Image of size 28x28 as in MNIST.
    """
    row_mean = tf.reduce_mean(x, axis=1)
    weights = np.concatenate([np.arange(1,15), np.arange(14, 0, -1)])
    weighted_intesity = row_mean * weights
    return np.mean(weighted_intesity)


def local_pixel_intensity_single(x, patch):
    # TODO: Find a way to accomodate single sample eval in the original function

    """Calc. avg. pixel intensity on local patch of image.

    x :: np.array
        dims = (28, 28).

    patch :: tuple[tuple]
        coordinates for patch. Tuple structure as follows: 
        ((x_dim_0, x_dim_1), (y_dim_0, y_dim_1)).

    """

    extracted_patch = x[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]]
    return tf.reduce_mean(extracted_patch)


def determine_holes(img, thresh=0.15):
    """Determine number of holes in image.
    
    img :: np.ndarray
        Mnist image.

    thresh :: float
        Threshhold value above which to set pixel values to 1 in binary image.
    """

    img_bin = img > thresh
    img_dil = ndimage.binary_dilation(img_bin)
    
    img_hole_diff = ndimage.binary_fill_holes(img_dil).astype(int) - img_dil.astype(int)
    img_hole_diff_alt = ndimage.binary_fill_holes(img_bin).astype(int) - img_bin.astype(int)
    
    
    img_hole_diff_ind = list(np.where(img_hole_diff > 0))
    img_hole_diff_alt_ind = list(np.where(img_hole_diff_alt > 0))
    
    img_hole_coord = list(zip(img_hole_diff_ind[0], img_hole_diff_ind[1]))
    img_hole_coord = [np.asarray(c) for c in img_hole_coord]
    
    img_hole_coord_alt = list(zip(img_hole_diff_alt_ind[0], img_hole_diff_alt_ind[1]))
    img_hole_coord_alt = [np.asarray(c) for c in img_hole_coord_alt]
    
    img_hole_count = 0
    img_hole_count_alt = 0
    
    for i in range(len(img_hole_coord) - 1):
        if i == 0: img_hole_count += 1
        
        e1 = img_hole_coord[i]
        e2 = img_hole_coord[i+1]

        if (abs(e2[0] - e1[0]) > 1) or ((abs(e2[0]- e1[0]) > 1) and (abs(e2[1] - e1[1]) > 1)):
            img_hole_count += 1
    
    for i in range(len(img_hole_coord_alt) - 1):
        if i == 0: img_hole_count_alt += 1

        e1 = img_hole_coord_alt[i]
        e2 = img_hole_coord_alt[i+1]

        if (abs(e2[0] - e1[0]) > 1) or ((abs(e2[0]- e1[0]) > 1) and (abs(e2[1] - e1[1]) > 1)):
            img_hole_count_alt += 1
            
    return (img_hole_count + img_hole_count_alt)//2


def modify_metric(metric, tag):
    """Add expected_input attribute to metric object and return it."""

    metric.expected_input = tag
    return metric


def random_experiment(confs, n_jobs=10):
    """Given a dictionary with configurations run random experiments.
    
    Parameters:
    confs :: dict
        Dictionary containing configurations for experiments.
        See sample dict below.

    n_jobs :: int
        Number of random experiments to be run.
    """
    hparam_choices = dict()
    confs_keys = list(confs.keys())

    for hparam, v in confs.items():
        select_indices = np.random.randint(0, len(v), size=n_jobs)
        hparam_choices[hparam] = v[select_indices]
    
    # bundle hparam_choices data together to obtain job configs
    hparam_unpacked =  list(zip(*hparam_choices.values()))
    jobs = [{confs_keys[i] : hparam_unpacked[j][i] for i in \
            range(len(confs_keys))} for j in range(len(hparam_unpacked))]

    return None


def run_and_eval_jobs(jobs):
    """Run a list of jobs and save results.
    
    Parameters:
    jobs :: list(dict)
        A list of dictionaries with each dictionary in the config format.

    """
    res_df = pd.DataFrame(columns=jobs[0].keys())
    res_df["results"] = None
    
    for job in jobs:

        results_job = run_config(job)
        job["results"] = results_job
        res_df = pd.concat([res_df, pd.Series(job).to_frame().T], 
                           ignore_index=True)
    
    return res_df


def run_config(conf, train_data, test_data, no_of_runs=10,
        eval_metrics=[modify_metric(tf.keras.metrics.BinaryAccuracy())]):
    """Run a job/ config and return results.
    
    Interpret config dictionary, build model, run training on data and
    evaluate model on test dataset. Do this no_of_runs times.

    Parameters:
    conf :: dict

    train_data :: tuple(np.ndarray, np.ndarray)

    test_data :: tuple(np.ndarray, np.ndarray)

    no_of_runs :: int
        Number of runs to make for same model to gauge variance of results.
    
    eval_metrics :: list[modify_metric(tf.keras.metrics object)]

    Returns:
    list of results for given metrics.
    
    """
    
    results = []

    for run in no_of_runs:    
        # repeat with same config no_of_runs times
        model = keras.Sequential([layers.Flatten(input_shape=(28,28))])

        # Parse conf dict

        # model architecture
        for layer_no in range(conf["model_arch"][0]):
            # add new layer with size as indicated in respective entry in list
            model.add(layers.Dense(conf["model_arch"][1][layer_no], activation="relu",
                                name="hl_"+str(layer_no)))
        
        model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))

        # parse no_of_predicates
        phi = phi_dct[conf["no_of_predicates"]]

        # parse model type
        # need no. of predicates for W
        w_matrix = tf.Variable(np.diag(np.ones(conf["no_of_predicates"])))

        if conf["model_type"] == "lusi":

            m = LusiModel(w_matrix, model=model)

        elif conf["model_type"] == "erm-lusi":
            m = LusiErm(w_matrix, conf["alpha"], model=model, erm_loss=tf.keras.losses.BinaryCrossentropy())
        
        else:
            # base model, compile
            m = ERM(model=model)


            model.compile(
                optimizer=keras.optimizers.SGD(),
                loss = keras.losses.BinaryCrossentropy(),
                metrics=eval_metrics
                )
        # parse total_data parameter
        train_data = lusi_periphery.get_data_excerpt(train_data, balanced=True,
                    size_of_excerpt=float(conf["total_data"]))
        
        periph = lusi_periphery.Periphery(train_data, test_data)

        # parse batch data
        train_batch = periph.generate_batch_data(
                                batch_size_1=conf["batch_size"][0],
                                batch_size_2=conf["batch_size"][1],
                                train_only=True)
        
        m.train(train_data, num_epochs=conf["num_epochs"], verbose=False)
        results.append(m.evaluate(test_data, eval_metrics))

    return results


config_dict = {
    "model_type" : ["lusi", "erm", "erm-lusi"],
    "model_arch" : [
                    (1, [50]), (1, [100]), (1, [500]), (2, [50, 20]),
                    (2, [100, 50]), (2, [500, 100]), (2, [1000, 500]),
                    (3, [100, 50, 20]), (3, [500, 200, 100]),
                    (3, [500, 100, 20]),
                    (4, [500, 300, 200, 100]), (4, [500, 200, 100, 50]),
                    (4, [500, 200, 50, 10]), (4, [100, 50, 20, 10])],
    "total_data" : [6000, 3000, 1000, 500, 200, 100, 64],
    "batch_size" : [(128, 128), (128, 64), (64, 64), (64, 32), (32, 64),
                     (32, 32), (32, 16), (16, 32), (32, 8), (8, 32), (8,8)],
    "no_of_predicates" : [3, 6, 9, 11],
    "epochs_training" : [1, 2, 5, 10, 15, 20, 30],
    "alpha" : [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]}


# partial func defs for use with Periphery class
local_pixel_intensity_center = functools.partial(local_pixel_intensity_single,
                                                 patch=((10,20), (10,20)))

local_pixel_intensity_ul = functools.partial(local_pixel_intensity_single,
                                             patch=((0,10), (0,10)))

local_pixel_intensity_ur = functools.partial(local_pixel_intensity_single,
                                             patch=((18,28), (0,10)))

local_pixel_intensity_ll = functools.partial(local_pixel_intensity_single,
                                             patch=((0,10), (18,28)))

local_pixel_intensity_lr = functools.partial(local_pixel_intensity_single,
                                             patch=((18,28), (18,28)))

symmetry_boxed_both_single = functools.partial(symmetry_boxed, single=True)

symmetry_boxed_vert_single = functools.partial(symmetry_boxed,
                                               axis="vertical", single=True)

symmetry_boxed_hor_single = functools.partial(symmetry_boxed,
                                              axis="horizontal", single=True)

determine_holes_15 = functools.partial(determine_holes, thresh=0.15)

phi_ls = [
    avg_pixel_intensity,
    symmetry_boxed_vert_single,
    symmetry_boxed_hor_single,
    symmetry_boxed_both_single,
    determine_holes_15,
    weighted_pixel_intesity,
    local_pixel_intensity_center,
    local_pixel_intensity_ll,
    local_pixel_intensity_ur,
    local_pixel_intensity_ul,
    local_pixel_intensity_lr
    ]

phi_3 = np.asarray(phi_ls[:3])
phi_6 = np.asarray(phi_ls[:6])
phi_9 = np.asarray(phi_ls[:9])
phi_11 = np.asarray(phi_ls)

phi_dct = {
    3 : phi_3,
    6 : phi_6,
    9 : phi_9,
    11 : phi_11
    }



# Specify some evaluation metrics for custom model
eval_metrics = [
    modify_metric(tf.keras.metrics.BinaryAccuracy(name="Binary Accuracy"),
                  "pred_and_true"),
    modify_metric(tf.keras.metrics.FalsePositives(name="False Positives"),
                  "pred_and_true"),
    modify_metric(tf.keras.metrics.FalseNegatives(name="False Negatives"),
                  "pred_and_true"),
    modify_metric(tf.keras.metrics.Precision(name="Precision"),
                  "pred_and_true"),
    modify_metric(tf.keras.metrics.Recall(name="Recall"),
                  "pred_and_true")
                # modify_metric(tf.keras.metrics.Mean(name="Mean"), "loss"),
                # modify_metric(tf.keras.metrics.Accuracy(), "pred_and_true")
    ]


def main():

    # create dataset
    # load mnist
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # extract 7 and 8
    eights = x_train[y_train == 8]/255
    sevens = x_train[y_train == 7]/255
    y_eights = np.ones(eights.shape[0])
    y_sevens = np.zeros(sevens.shape[0])
    # merge 7 and 8
    x_train = np.concatenate([eights, sevens])
    y_train = np.concatenate([y_eights, y_sevens])

    # same for test data
    eights_test = x_test[y_test == 8]/255
    sevens_test = x_test[y_test == 7]/255
    y_eights_test = np.ones(eights_test.shape[0])
    y_sevens_test = np.zeros(sevens_test.shape[0])
    x_test = np.concatenate([eights_test, sevens_test])
    y_test = np.concatenate([y_eights_test, y_sevens_test])

    # data = lusi_periphery.Periphery((x_train, y_train), (x_test, y_test), phi)
    data = lusi_periphery.Periphery(lusi_periphery.get_data_excerpt(
        (x_train, y_train), size_of_excerpt=0.053),
        (x_test, y_test), phi_11)
    train_batch, test_batch = data.generate_batch_data(64,64)
    
    # create lusi model
    w_matrix = tf.Variable(np.diag(np.ones(11)), dtype=tf.float32)
    lusi_model = LusiModel(w_matrix)
    lusi_model.add_optimizer(tf.keras.optimizers.SGD())
    res_b_train = lusi_model.evaluate(data.test_data, eval_metrics)
    lusi_model.train(train_batch, num_epochs=10)
    res_a_train = lusi_model.evaluate(data.test_data, eval_metrics)
    
    # create lusi-erm model, 0.8 weighting for erm, 0.2 for LUSI
    lusi_erm = LusiErm(w_matrix, alpha=0.8)
    lusi_erm.add_optimizer(tf.keras.optimizers.SGD())
    res_b_train_lusi_erm = lusi_erm.evaluate(data.test_data, eval_metrics)
    lusi_erm.train(train_batch, num_epochs=10)
    res_a_train_lusi_erm = lusi_erm.evaluate(data.test_data, eval_metrics)

    # create baseline model
    baseline_bin_class = keras.Sequential(
    [
    layers.Flatten(input_shape=(28,28)),
    # layers.Dense(100, activation="relu", name="hidden_layer_1"),
    layers.Dense(500, activation="relu", name="hidden_layer_2"),
    layers.Dense(1, activation="sigmoid", name="output_layer") # interpret output as prob. for class 1
    # layers.Dense(1, name="output_layer", activation="relu")
    ])
    
    baseline_bin_class.compile(
    optimizer=keras.optimizers.SGD(),
    # loss=keras.losses.SparseCategoricalCrossentropy(),
    # loss=keras.losses.binary_crossentropy(),
    loss = keras.losses.BinaryCrossentropy(),
    # metrics=[keras.metrics.BinaryAccuracy(), "accuracy"]
    metrics=eval_metrics
    )

    res_b_train_base = baseline_bin_class.evaluate(x_test, y_test,
        batch_size=x_test.shape[0])

    baseline_bin_class.fit(data.train_data_x, data.train_data_y,
        batch_size=64, epochs=10)
    
    res_a_train_base = baseline_bin_class.evaluate(x_test, y_test,
        batch_size=x_test.shape[0])
    perf_summary = {
        "LUSI": (res_b_train, res_a_train),
        "ERM" : (res_b_train_base, res_a_train_base),
        "LUSI-ERM" : (res_b_train_lusi_erm, res_a_train_lusi_erm)}
    
    return None

if __name__ == "__main__":
    main()
