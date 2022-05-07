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

class LusiLossNoBatch(tf.losses.Loss):


    def __init__(self, phi_x, tau, W):
        """Initialize Loss object given predicates and weight matrix.

        Assume that predicates are factorizing.
        
        Parameters:
            phi_x :: tensor
            array of evaluations of predicates over space X
            dims: (batch_size, no_of_predicates)
            
            tau :: np.array
            array of predicates over Y space.
            
            W :: tensor
            symmetric, positive definite weight matrix of adequate dimensions.
        """
        
        # Check if dims are ok
        if not (tau.shape[0] == W.shape[0] and W.shape[0] == W.shape[1]):
            raise Exception("Check dims tau and W.")

        super().__init__()
        self.phi_x = phi_x
        self.tau = tau
        self.W = W
    

    def call(self, y_pred, y_true):
        # y_pred and y_true expected to have dims (batch_size, d0, ..., dN) where di means dimension i.
        # Explicitly for bin. class - dims are given by (batch_size, d0)
        
        if not y_true.shape[0] == y_pred.shape[0]:
            raise Exception("Check dims y-values.")
        
        if not self.phi_x.shape[1] == self.tau.shape[0]:
            raise Exception("Check dims tau and phi_x")
        
        if not self.phi_x.shape[0] == y_pred.shape[0]:
            raise Exception("Check dims y and phi_x")
        
        y_dim = y_true.shape[0]

        # for each y_true, calc vector [tau_1(y_true), ..., tau_d(y_true)], same for y_pred
        y_tau = [self.tau[i](y_true[j]) for j in range(y_true.shape[0])  for i in range(self.tau.shape[0])]        
        y_true = tf.convert_to_tensor(y_tau, dtype=tf.float64)
        y_true = tf.reshape(y_true, [y_dim, self.tau.shape[0]])

        y_pred_tau = [self.tau[i](y_pred[j]) for j in range(y_pred.shape[0])  for i in range(self.tau.shape[0])]
        y_pred = tf.convert_to_tensor(y_pred_tau, dtype=tf.float64)
        y_pred = tf.reshape(y_pred,[y_dim, self.tau.shape[0]])
        

        
        no_weight_loss = tf.reduce_mean(self.phi_x * (y_true - y_pred), axis=0)
        # no_weight_loss should be tensor of dims (1, no_of_predicates)
        no_weight_loss = tf.reshape(no_weight_loss, [self.tau.shape[0],1])

        return dot(tf.transpose(no_weight_loss), no_weight_loss, self.W)


class LusiModel(tf.keras.Model):
    """Given a model, e.g. neural net, implement the LUSI approach."""
    
    def __init__(self, m_inner_prod, model=None, tau=None) -> None:
        """Instatiate model with custom training loop adapted to LUSI method.

        Parameters:

        m_inner_prod :: tf.Variable with appropriate dims.
        Matrix used for custom inner product in Lusi Loss ('M' in paper).

        model :: tensorflow model or None
        Underlying model to be used to make predictions.

        tau :: 
        Predicates of the form f : \mathcal{Y} -> \R.
        See paper section 'factorizing predicates'.

        """

        super().__init__()
        if not model:
            # if no model given, use this small feedforward neural net as default
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


    def summary(self):
        return self.model.summary()

    
    def add_optimizer(self, optimizer) -> None:
        """Add tf optimizer to use."""
        self.optimizer = optimizer


    def add_metrics(self, metrics) -> None:
        pass


    def train(self, train_dataset, num_epochs, batch_1_size,
              train_metrics : list=[]):
        """Training loop.
    
        Parameters:

        train_dataset :: BatchDataSet
        Must include predicates evaluated, x, y values.
    
        ...
        """
        assert self.optimizer, "No optimizer specified."
        
        epoch_train_metrics_results = []

        for epoch in range(num_epochs):
            print("\nStart of epoch %d" % (epoch,))   


            # Iterate over the batches of the dataset.
            for step, (pred_batch, x_batch_train, y_batch_train) in enumerate(train_dataset):
            
                x_J = x_batch_train[:batch_1_size]
                x_J_prime = x_batch_train[batch_1_size:]
                y_J_true = y_batch_train[:batch_1_size]
                y_J_prime_true = y_batch_train[batch_1_size:]
                
                # Dirty solution -> one type for every tensor in advance
                y_J_prime_true = tf.cast(y_J_prime_true, tf.float32)
                y_J_prime_true = tf.expand_dims(y_J_prime_true, axis=1)
                
                
                pred_J = pred_batch[:batch_1_size]
                pred_J_prime = pred_batch[batch_1_size:]
                
                
                
                y_J_prime_pred = self.model(x_J_prime)
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    
                    y_J_pred = self.model(x_J, training=True)  # Logits for J batch recorded on gradient tape
                    # tape.watch(y_J_pred)
                    # y_J_test = k_bin_class(x_J, training=True)  # Logits for J batch recorded on gradient tape
                    # y_pred = tf.concat([y_J_pred, y_J_prime_pred], axis=0)  # not sure if needed
                    
                    y_J_pred = tf.broadcast_to(y_J_pred, shape=[y_J_pred.shape[0], pred_J.shape[1]])
                    
                    # Compute the loss value for this minibatch.
                    v = tf.reduce_mean(pred_J * y_J_pred, axis=0, keepdims=True)

                    v_prime_inter = tf.broadcast_to(y_J_prime_pred - y_J_prime_true, 
                                                    shape=[y_J_prime_true.shape[0], pred_J_prime.shape[1]])

                    v_prime = tf.reduce_mean(pred_J_prime * v_prime_inter, axis=0, keepdims=True)
                    
                    
                    v_prime_times_weight_matrix = tf.matmul(self.m_inner_prod, tf.transpose(v_prime))

                    loss_value = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                            tf.tensordot(v, tf.matmul(self.m_inner_prod, tf.transpose(v_prime)), axes=1))
                    
                # watched_vars = tape.watched_variables()
                # Use the gradient tape to automatically retrieve
                
                # the gradients of the trainable variables with respect to the loss.
                # grads = tape.gradient(y_J_test, k_bin_class.trainable_weights)
                # grads_list.append(grads)
                for eval_metric in train_metrics:
                    if eval_metric.expected_input == "pred_and_true":
                        eval_metric.update_state(y_batch_train, tf.round(self.model(x_batch_train, training=True)))
                    elif eval_metric.expected_input == "loss":
                        eval_metric.update_state(loss_value)
                
                # debugging only - add metric scores before first update
                if epoch == 0 and step==0:
                    epoch_train_metrics_results.append(
                    [(eval_metric.name, eval_metric.result().numpy()) for eval_metric in train_metrics]
                    )  


                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # TODO grads list
                # grads_list.append(grads)
                # Until here, seems to work fine.
                
                
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                # print(k_bin_class.trainable_weights)
                # print(grads)
                

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                # print(k_bin_class.trainable_weights)
                # raise Exception
                # TODO trainable weights list
                # trainable_weights_list.append(k_bin_class.trainable_weights)
            
                # Log every 200 batches.
                if step % 100 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    # TODO: instead of 64 use batch size
                    print("Seen so far: %s samples" % ((step + 1) * 64))

            # Epoch stats
            epoch_train_metrics_results.append(
                [(eval_metric.name, eval_metric.result().numpy()) for eval_metric in train_metrics]
            )

            # reset metrics
            for eval_metric in train_metrics:
                eval_metric.reset_state()
            
        
        self.epoch_train_metrics_results = epoch_train_metrics_results
        
        return None


    def train_debug(self, train_dataset, num_epochs, batch_1_size,
              train_metrics : list=[]):
        """Training loop.
    
        Parameters:

        train_dataset :: BatchDataSet
        Must include predicates evaluated, x, y values.
    
        ...
        """
        assert self.optimizer, "No optimizer specified."
        gradient_list = []
        epoch_train_metrics_results = []
        model_weight_list = []

        model_weight_list.append((-1,[r.numpy() for r in self.model.trainable_weights]))

        for epoch in range(num_epochs):
            print("\nStart of epoch %d" % (epoch,))   


            # Iterate over the batches of the dataset.
            for step, (pred_batch, x_batch_train, y_batch_train) in enumerate(train_dataset):
                # J = np.random(B)
                # J' = np.random(B')
                
                x_J = x_batch_train[:batch_1_size]
                x_J_prime = x_batch_train[batch_1_size:]
                y_J_true = y_batch_train[:batch_1_size]
                y_J_prime_true = y_batch_train[batch_1_size:]
                
                # Dirty solution -> one type for every tensor in advance
                y_J_prime_true = tf.cast(y_J_prime_true, tf.float32)
                y_J_prime_true = tf.expand_dims(y_J_prime_true, axis=1)
                
                
                pred_J = pred_batch[:batch_1_size]
                pred_J_prime = pred_batch[batch_1_size:]
                
                
                
                y_J_prime_pred = self.model(x_J_prime)
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    
                    y_J_pred = self.model(x_J, training=True)  # Logits for J batch recorded on gradient tape
                    # tape.watch(y_J_pred)
                    # y_J_test = k_bin_class(x_J, training=True)  # Logits for J batch recorded on gradient tape
                    # y_pred = tf.concat([y_J_pred, y_J_prime_pred], axis=0)  # not sure if needed
                    
                    y_J_pred = tf.broadcast_to(y_J_pred, shape=[y_J_pred.shape[0], pred_J.shape[1]])
                    
                    # Compute the loss value for this minibatch.
                    v = tf.reduce_mean(pred_J * y_J_pred, axis=0, keepdims=True)

                    v_prime_inter = tf.broadcast_to(y_J_prime_pred - y_J_prime_true, 
                                                    shape=[y_J_prime_true.shape[0], pred_J_prime.shape[1]])

                    v_prime = tf.reduce_mean(pred_J_prime * v_prime_inter, axis=0, keepdims=True)
                    
                    
                    v_prime_times_weight_matrix = tf.matmul(self.m_inner_prod, tf.transpose(v_prime))

                    loss_value = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                            tf.tensordot(v, tf.matmul(self.m_inner_prod, tf.transpose(v_prime)), axes=1))
                    
                watched_vars = tape.watched_variables()
                # Use the gradient tape to automatically retrieve
                
                # the gradients of the trainable variables with respect to the loss.
                # grads = tape.gradient(y_J_test, k_bin_class.trainable_weights)
                # grads_list.append(grads)
                for eval_metric in train_metrics:
                    if eval_metric.expected_input == "pred_and_true":
                        eval_metric.update_state(y_batch_train, tf.round(self.model(x_batch_train, training=True)))
                    elif eval_metric.expected_input == "loss":
                        eval_metric.update_state(loss_value)
                
                # debugging only - add metric scores before first update
                if epoch == 0 and step==0:
                    epoch_train_metrics_results.append(
                    [(eval_metric.name, eval_metric.result().numpy()) for eval_metric in train_metrics]
                    )  


                grads = tape.gradient(loss_value, self.model.trainable_weights)
                gradient_list.append(((epoch, step), grads))
            
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                model_weight_list.append(((epoch, step), [r.numpy() for r in self.model.trainable_weights]))
                
            
                # Log every 200 batches.
                if step % 100 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    # TODO: instead of 64 use batch size
                    print("Seen so far: %s samples" % ((step + 1) * 64))

            # Epoch stats
            epoch_train_metrics_results.append(
                [(eval_metric.name, eval_metric.result().numpy()) for eval_metric in train_metrics]
            )

            # reset metrics
            for eval_metric in train_metrics:
                eval_metric.reset_state()
            
        
        self.epoch_train_metrics_results = epoch_train_metrics_results
        self.gradient_list = gradient_list
        self.model_weight_list = model_weight_list
        
        return None


    def train_correct(self, train_dataset, num_epochs, train_metrics : list=[]):
        """Training loop.
    
        Parameters:

        train_dataset :: zipped BatchDataSet
        Must include predicates evaluated, x, y values.
    
        ...
        """


        assert self.optimizer, "No optimizer specified."

        gradient_list = []
        epoch_train_metrics_results = []
        model_weight_list = []

        # model_weight_list.append((-1,[r.numpy() for r in self.model.trainable_weights]))
        model_weight_list.append((-1, self.layers[0].get_weights()))
        
        for epoch in range(num_epochs):
            print("\nStart of epoch %d" % (epoch,))   


            # Iterate over the batches of the dataset.
            for step, ((pred_batch_1, x_batch_1, y_batch_1), 
                (pred_batch_2, x_batch_2, y_batch_2))  \
                    in enumerate(train_dataset):
                
                
                # need total batch for evaluation and calc of
                # 'true' loss.

                x_total = tf.concat([x_batch_1, x_batch_2], axis=0)  # not sure if needed
                
                y_total = tf.concat([y_batch_1, y_batch_2], axis=0)
                
                # Dirty solution -> one type for every tensor in advance
                y_batch_2 = tf.cast(y_batch_2, tf.float32)
                y_batch_2= tf.expand_dims(y_batch_2, axis=1)
                
                # Predictions on 2nd batch. This calc should not be recorded
                # on gradient tape
                y_batch_2_pred = self.model(x_batch_2)

                # broadcasting difference with actual labels to shape which
                # is compatible for multiplication with predicates
                # result: d identical columns where d = no. of predicates.
                # v_prime_inter = tf.broadcast_to(y_batch_2_pred - y_batch_2,
                #                                 shape=[y_batch_2.shape[0],
                #                                 pred_batch_2.shape[1]])

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Logits for batch J recorded on gradient tape 
                    y_batch_1_pred = self.model(x_batch_1, training=True)
                    
                    # prepare for multipication with predicate evaluations
                    # result: d identical columns where d = no. of predicates.
                    y_batch_1_pred = tf.broadcast_to(y_batch_1_pred,
                        shape=[y_batch_1_pred.shape[0], pred_batch_1.shape[1]])
                    
                    # Compute the loss value to be differentiated for batch.
                    v = tf.reduce_mean(pred_batch_1 * y_batch_1_pred, axis=0, keepdims=True)

                    v_prime_inter = tf.broadcast_to(y_batch_2_pred - y_batch_2, 
                                                    shape=[y_batch_2.shape[0],
                                                    pred_batch_2.shape[1]])

                    v_prime = tf.reduce_mean(pred_batch_2 * v_prime_inter,
                                            axis=0, keepdims=True)
                    
                    
                    # v_prime_times_weight_matrix = tf.matmul(self.m_inner_prod, tf.transpose(v_prime))

                    loss_value = tf.multiply(tf.Variable(2, dtype=tf.float32),
                                            tf.tensordot(v, tf.matmul(self.m_inner_prod, tf.transpose(v_prime)), axes=1))
                    
                    actual_loss = None  # actual loss calc here
                    
                watched_vars = tape.watched_variables()
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                
                for eval_metric in train_metrics:
                    if eval_metric.expected_input == "pred_and_true":
                        eval_metric.update_state(y_total, tf.round(self.model(x_total, training=True)))
                    elif eval_metric.expected_input == "loss":
                        eval_metric.update_state(loss_value)
                
                # debugging only - add metric scores before first update
                if epoch == 0 and step==0:
                    epoch_train_metrics_results.append(
                    [(eval_metric.name, eval_metric.result().numpy()) for eval_metric in train_metrics]
                    )  

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                gradient_list.append(((epoch, step), grads))
            
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                # model_weight_list.append(((epoch, step), [r.numpy() for r in self.model.trainable_weights]))
                model_weight_list.append(((epoch, step), self.layers[0].get_weights()))

                # Log every 200 batches.
                if step % 100 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    # TODO: instead of 64 use batch size
                    print("Seen so far: %s samples" % ((step + 1) * 64))

            # Epoch stats
            epoch_train_metrics_results.append(
                [(eval_metric.name, eval_metric.result().numpy()) for eval_metric in train_metrics]
            )

            # reset metrics
            for eval_metric in train_metrics:
                eval_metric.reset_state()
            
        
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


def main():
    # bin_class = keras.Sequential(
    # [
    #     layers.Flatten(input_shape=(28,28)),
    #     layers.Dense(700, activation="relu", name="hidden_layer_01"),
    #     layers.Dense(1, name="output_layer", activation="sigmoid") # interpret output as prob. for class 1
    # ]
    # )
    preds = np.mean
    weight_matrix = np.diag(np.ones(3))
    m = LusiModel(preds, weight_matrix, model=None)
    m.summary()

if __name__ == "__main__":
    main()

phi = np.asarray([avg_pixel_intensity, weighted_pixel_intesity, local_pixel_intensity_center])

