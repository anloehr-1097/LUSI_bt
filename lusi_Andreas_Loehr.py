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
        





def symmetry(imgs, axis="both"):
    """Calculate a symmetry score for normalized pictures.
    
    Parameters:
    img :: np.array
        Numpy array representing images. Pixel values in range [0,1]
    
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


def symmetry_boxed(imgs):
    """Determine symmetry of cropped images.
    
    Assume [0,1]-range for pixels.
    """
    
    cropped_imgs = determine_box(imgs)
    sym_scores = np.zeros(len(cropped_imgs))
    
    for j in range(len(imgs)):
        sym_scores[j] = symmetry(cropped_imgs[j])
    
    return sym_scores


def dot(a, b, W):
    """Calculate dot product between a, b using W as weight matrix.
    
    Parameters:

    a :: tensor of dtype tf.float64
    transposed vector
    
    b :: tensor of dtype tf.float64
    non-transposed vector
    
    W :: tensor of dtype tf.float64
    matrix.

    Returns:
    dot product induced by W of vectors a and b.
    """
    
    return tf.tensordot(a, tf.matmul(W, b), 1)  


def weighted_pixel_intesity(x):
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
    coordinates for patch. Tuple structure as follows: ((x_dim_0, x_dim_1), (y_dim_0, y_dim_1)).

    """

    extracted_patch = x[patch[0][0]:patch[0][1], patch[1][0]: patch[1][1]]
    return tf.reduce_mean(extracted_patch)


def apply_predicates_on_data(predicates, x):
    """ Evaluate predicates on data and store values.
    
    predicates :: np.array
    List of \R valued predicates working on x of dimension d.
    
    x :: np.array
    data to apply predicates to of dimensinos (n, d0, ..., dN).
    I.e. for MNIST, dims = (n, 28, 28)
    
    returns:
    array of dimensions (n, d).
    """
    
    pred_evals = np.asarray([predicates[i](x[j]) for j in range(x.shape[0]) for i in range(predicates.shape[0])])
    pred_evals = pred_evals.reshape(x.shape[0], predicates.shape[0])
    pred_evals = tf.convert_to_tensor(pred_evals, dtype=tf.float32)
    
    return pred_evals    


local_pixel_intensity_center = functools.partial(local_pixel_intensity_single, patch=((10,20), (10,20)))


def modify_metric(metric, tag):
    """Add expected_input attribute to metric object and return it."""

    metric.expected_input = tag
    return metric





phi = np.asarray([avg_pixel_intensity, weighted_pixel_intesity, local_pixel_intensity_center])
# Specify some evaluation metrics for custom model
eval_metrics = [modify_metric(tf.keras.metrics.BinaryAccuracy(name="Binary Accuracy"), "pred_and_true"), 
                modify_metric(tf.keras.metrics.FalsePositives(name="False Positives"), "pred_and_true"), 
                modify_metric(tf.keras.metrics.FalseNegatives(name="False Negatives"), "pred_and_true"), 
                modify_metric(tf.keras.metrics.Precision(name="Precision"), "pred_and_true"), 
                modify_metric(tf.keras.metrics.Recall(name="Recall"), "pred_and_true"),
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

    data = lusi_periphery.Periphery((x_train, y_train), (x_test, y_test), phi)
    train_batch, test_batch = data.generate_batch_data(54,32)
     
    # create model
    w_matrix = tf.Variable(np.diag(np.ones(3)), dtype=tf.float32)
    lusi_model = LusiModel(w_matrix)
    lusi_model.add_optimizer(tf.keras.optimizers.SGD())
    res_b_train = lusi_model.evaluate(data.test_data, eval_metrics)
    lusi_model.train(train_batch, num_epochs=2)
    res_a_train = lusi_model.evaluate(data.test_data, eval_metrics)
    
    return None

if __name__ == "__main__":
    main()
