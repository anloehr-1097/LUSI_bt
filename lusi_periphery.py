import tensorflow as tf
import numpy as np
import functools
import itertools
import re
import matplotlib.pyplot as plt
from primefac import primefac
from pprint import pprint
import pandas as pd

class Periphery:
    def __init__(self, train_data=None, test_data=None, phi=np.array([])):
        
        """Bind all data and objetcs for Lusi and ERM-Lusi method.
        
        Parameters:
        train_data :: tuple of np.ndarrays (x_train, y_train) | None
            Training data.
        
        test_data :: tuple of np.ndarrays (x_test, y_test) | None
            Test data.
            
        phi :: np.ndarray[list[function]]
            A numpy array of list of functions to be applied on x_data.      
        """
        
        # create attributes with None values, will be populated in set_phi method
        self.phi = phi
        self.phi_eval_train = tf.constant([])
        self.phi_eval_test = tf.constant([])
        self.train_data = None
        self.test_data = None
        
        self.set_phi(phi)
        self.set_train_data(train_data)
        self.set_test_data(test_data)

        return None
    

    def generate_batch_data(self, batch_size_1=64, batch_size_2=64, train_only=False):
        """Generate batch datasets for training data (and test data).
        
        Parameters:
        
        batch_size_1 :: int
            See paper for meaning of batch_size_1 denoted as B.
        
        batch_size_2 :: int
            See paper for meaning of batch_size_1 denoted as B'.
            
        train_only :: Bool
            Create batch dataset for train data only.
        
        Returns:
        A tuple of batched datasets. If train_only, 1-tuple with batched
        train dataset, else, batched train and batched test datasets.
        """
        
        # Check for provision of relevant data.
        # If not provided until call of this method, raise error.
        if not self.train_data:
            raise ValueError("No training data provided.")
            
        if not (self.test_data or train_only):
            raise ValueError("No test data provided.")
            
        if not (tf.size(self.phi_eval_train) > 0).numpy():
            # requires evaluation of phis on training data
            raise ValueError("No phi evaluation data for training data provided.")
        
        #
        dataset_train = tf.data.Dataset.from_tensor_slices(
            (self.phi_eval_train, self.train_data[0], self.train_data[1]))
        
        # Important: need drop remainder for training to work
        # (was a problem, not sure if still persists)
        batch_dataset_1_train = dataset_train.shuffle(
            buffer_size=1024).batch(batch_size_1, drop_remainder=True)

        batch_dataset_2_train = dataset_train.shuffle(
            buffer_size=1024).batch(batch_size_2, drop_remainder=True)

        batch_dataset_train = tf.data.Dataset.zip((batch_dataset_1_train,
                                                   batch_dataset_2_train))
        
        if train_only:
            return (batch_dataset_train, )
        
        
        # also generate batch dataset for test data
        if not (tf.size(self.phi_eval_test) > 0).numpy():
            raise ValueError("No phi evaluation data for test data provided.")
        
        dataset_test = tf.data.Dataset.from_tensor_slices(
            (self.phi_eval_test, self.test_data[0], self.test_data[1]))

        batch_dataset_1_test = dataset_test.shuffle(
            buffer_size=1024).batch(batch_size_1, drop_remainder=True)
        
        batch_dataset_2_test = dataset_test.shuffle(
            buffer_size=1024).batch(batch_size_2, drop_remainder=True)

        batch_dataset_test = tf.data.Dataset.zip((batch_dataset_1_test,
                                                  batch_dataset_2_test))
            
        return (batch_dataset_train, batch_dataset_test)
   

    def apply_phi(self, which_dataset="both"):
        """ Evaluate predicates (phi : \mathcal{X} -> \R) on data & store.

        Parameters:

        which_dataset :: str
            One of "train", "test", "both".
            Indicates which dataset phi should be applied to.

        """

        if which_dataset == "both":
            if not (self.train_data and self.test_data):
                raise ValueError("Some data missing. Check if both train and \
                                  test data provided.")

            train_x = self.train_data[0]
            test_x = self.test_data[0]
            
            phi_x_train = np.asarray([self.phi[i](train_x[j]) for j in \
                range(train_x.shape[0]) for i in \
                range(self.phi.shape[0])], dtype="object")

            phi_x_test = np.asarray([self.phi[i](test_x[j]) for j in \
                range(test_x.shape[0]) for i in \
                range(self.phi.shape[0])], dtype="object")

            phi_x_train = phi_x_train.reshape(train_x.shape[0],
                                              self.phi.shape[0])
            
            phi_x_test = phi_x_test.reshape(test_x.shape[0],
                                            self.phi.shape[0])

            phi_x_train = tf.convert_to_tensor(phi_x_train, dtype=tf.float32)
            phi_x_test = tf.convert_to_tensor(phi_x_test, dtype=tf.float32)

            self.phi_eval_train = phi_x_train
            self.phi_eval_test = phi_x_test

        elif which_dataset == "train":

            train_x = self.train_data[0]
            phi_x_train = np.asarray([self.phi[i](train_x[j]) for j in \
                range(train_x.shape[0]) for i in \
                range(self.phi.shape[0])], dtype="object")
            
            phi_x_train = phi_x_train.reshape(train_x.shape[0],
                                              self.phi.shape[0])

            phi_x_train = tf.convert_to_tensor(phi_x_train, dtype=tf.float32)
            self.phi_eval_train = phi_x_train

        else:
            test_x = self.test_data[0]
            phi_x_test = np.asarray([self.phi[i](test_x[j]) for j in \
                range(test_x.shape[0]) for i in \
                range(self.phi.shape[0])], dtype="object")

            phi_x_test = phi_x_test.reshape(test_x.shape[0],
                                            self.phi.shape[0])

            phi_x_test = tf.convert_to_tensor(phi_x_test, dtype=tf.float32)
            self.phi_eval_test = phi_x_test
            
        return None
    
        
    def set_phi(self, phi):
        """Set phi as attribute to object, call apply_phi method.
        
        Parameters:
        
        phi :: np.ndarray[functions]
            An array of functions to be applied on data. No checks are
            implemented checking if phi is defined on input data shape or not.
            Has only been tested for the predicates in the module
            'lusi_AndreasLoehr.py' which operate on images of MNIST
            dataset.
        """
        
        if not phi.size > 0:
            # only apply if phi exist
            return None
        
        self.phi = phi
        
        # auto infer on which data to apply phi
        if self.train_data:
            if self.test_data:            
                self.apply_phi("both")
            else:
                self.apply_phi("train")
        
        else:
            if self.test_data:
                self.apply_phi("test")

        return None
    
    
    def set_train_data(self, train_data):
        """Set train_data as attribute to object.
        
        Parameters:
        
        train_data :: tuple(np.ndarray) | None
            Tuple (x_train, y_train) of train data where each element is a
            np.ndarray
        """

        if not train_data:
            return None
            
        
        self.train_data = train_data
        self.train_data_x = train_data[0]
        self.train_data_y = train_data[1]
        
        if self.phi.size > 0:
            self.apply_phi("train")

        return None
    
    
    def set_test_data(self, test_data):
        """Set train_data as attribute to object.
        
        Parameters:
        
        test_data :: tuple(np.ndarray) | None
            Tuple (x_test, y_test) of test data where each element is a
            np.ndarray
        """

        if not test_data:
            return None
        
        self.test_data = test_data
        self.test_data_x = test_data[0]
        self.test_data_y = test_data[1]
         
        if self.phi.size > 0:        
            self.apply_phi("test")

        return None



def get_data_excerpt(data, balanced=False, size_of_excerpt=0.5):
    """Get portion of passed data.
    
    This method is assuming a more or less balanced dataset to start with.

    Parameters:
    data :: tuple of np.ndarrays
        Consists of ndarrays for x and y.
        Assumes y made up of 2 classes.
    
    balanced :: bool
        Determines if classes should be balanced.

    size_of_excerpt :: [int | float]
        Size of excerpt to be extracted from data.
        If int is passed, then intepreted as abs. number of samples.
        If float is passed, then interpreted as relative portion of data.

    Returns:
    tuple (x_exc, y_exc) of np.ndarrays.
    """
    
    if isinstance(size_of_excerpt, float) and size_of_excerpt <= 1:
        size_of_excerpt = np.floor(
            size_of_excerpt * data[0].shape[0]).astype(int)
    
    if not size_of_excerpt <= data[0].shape[0]:
        raise IndexError("Sample size too large.")
    
    # get random indices
    permute = np.random.permutation(np.arange(size_of_excerpt)).astype(int)

    if balanced:

        y_cls = set(data[1])
        # print(f"No. of classes = {len(y_cls)}")
        if len(y_cls) != 2:
            raise Exception("Pass data for binary classification problem.")
        
        y_1_cls = y_cls.pop()
        y_2_cls = y_cls.pop()
        
        x_1 = data[0][data[1] == y_1_cls]
        x_2 = data[0][data[1] == y_2_cls]

        y_1 = data[1][data[1] == y_1_cls]
        y_2 = data[1][data[1] == y_2_cls]
        ind_1 = np.random.randint(low=0, high=x_1.shape[0],
                                  size=int(size_of_excerpt//2))
                        
        ind_2 = np.random.randint(low=0, high=x_2.shape[0],
                                          size=int((size_of_excerpt - \
                                                    size_of_excerpt//2)))
        
        x = np.concatenate([x_1[ind_1], x_2[ind_2]])
        y = np.concatenate([y_1[ind_1], y_2[ind_2]])


        if not permute.shape[0] == x.shape[0]:
            raise Exception("Something has been lost on the way...")

        return (x[permute],y[permute])
    
    else:
        ind= np.random.randint(low=0, high=data[0].shape[0],
                                       size=size_of_excerpt)
        x = data[0][ind]
        y = data[1][ind]

        return (x[permute],y[permute])


def visual_validation(n, data):
    """Plot n samples with their predicitions against true labels.
    
    n :: int
        Number of samples to be plotted.
    
    data :: tuple of form (x, y, y_pred)
        Tuple should consist of np.ndarrays for x, y, and model predictions.

    """
    # TODO: also plot if pred not provided.
    # determine prime factors of n to determine 'acceptable' layout
    pfs = list(primefac(n))
    three_quart = np.floor(len(pfs) * 0.75).astype(int)
    rowcount = np.prod(pfs[: three_quart])
    colcount = np.prod(pfs[three_quart :])
    
    # select n datapoints from data, bring into grid shape for easy handling
    random_indices = np.random.randint(0, high=data[0].shape[0], size=n)
    random_indices = np.reshape(random_indices, (rowcount, colcount))
    
    # determine size of figure dynamically
    size = np.max([colcount, rowcount]) * 3
    fig, ax = plt.subplots(nrows=rowcount, ncols=colcount,
                           figsize=(size, size), constrained_layout=True)
    if len(data) == 3:
    # populate grid with images + labels
        for i in range(rowcount):
            for j in range(colcount):
                ax[i,j].imshow(data[0][random_indices[i,j]])
                ax[i,j].set_title(
                    "pred: ({:.2f} -> {:.0f}),  true: {:.0f}".format(\
                    data[2][random_indices[i,j]][0],
                    np.round(data[2][random_indices[i,j]][0]),
                    data[1][random_indices[i,j]]))
                # ax[i,j].set_title(f"pred: {data[2][random_indices[i,j]][0]}, 
                # true: {data[1][random_indices[i,j]]}")
                ax[i,j].axis("off")

    else:
        # assume data has 2 entries (x, y)
        for i in range(rowcount):
            for j in range(colcount):
                ax[i,j].imshow(data[0][random_indices[i,j]])
                ax[i,j].set_title(
                    "Label: {:.0f}".format(data[1][random_indices[i,j]]))
                ax[i,j].axis("off")

    return None


def parse_model_arch(arch):
    """Parse model architecture entry from result dataframe.
    
    Parameters:

    arch :: str
        architecture string of the form '[d, list(d, d, d)]' where d is numeral.
    
    Returns:

    """
    
    layers_re = r"[\d]+(?= list)"
    width_re = r"(?<=list\(\[)[\d ,]+"
    model_layer_list = re.findall(layers_re, arch)[0].split(", ")
    model_width_list = re.findall(width_re, arch)[0].split(", ")
    widths = [int(d) for d in model_width_list]
    layers = [int(l) for l in model_layer_list]
    
    if not len(layers) == 1:
        raise Exception(f"Layers should be of int type. Instead got the value {layers}")
    
    layers = layers[0]
    
    return layers, widths


def parse_parsed_arch(parsed_arch):
    """Extract widths and depth of network form parsed_arch.
    
    Parameters:

    parsed_arch ::  pd.Series
        return from parse_model_arch function applied to entire column via
        df method.
    
    Returns: tuple containing 2 pd.Series, one for no. of layers,
             one for list of layers' breadths.
    """
    
    depths = parsed_arch.apply(lambda x: x[0])
    widths = parsed_arch.apply(lambda x: x[1])
    
    return depths, widths
    

def parse_batch_size(bs):
    """"Parse batch size column from testing result_dataframe.
   
    Parameters:

    bs :: str
        string of the form '[d, d]' where d is numeral.

    Returns:
    List containing batch size 1 and batch size 2.   
    """
    
    bs = bs.replace("[", "").replace("]", "").split(" ")
    bs_list = [int(s) for s in bs]
    return bs_list


def parse_res_def(df):
    """Parse results dataframe s.t. it is easier to process.
    
    df :: pd.DataFrame
        Dataframe as obtained by the function run_and_eval_jobs in lusi_Andreas_Loehr.
    
    Returns: 
    """
    # from col seven of original df onwards acc scores
    df["avg_bin_acc"] = df.to_numpy()[:, 7:].mean(axis=1)

    model_arch = df["model_arch"].apply(parse_model_arch)
    depths, widths = parse_parsed_arch(model_arch)
    df["layers"] = depths
    
    max_depth = 10
    width_ar = np.zeros(len(widths)*max_depth).reshape(len(widths), max_depth)
    
    for i, width in enumerate(widths):
        # print(f"i = {i}, width={width}")
        for j, d in enumerate(width):
            # print(f"j = {j}, d={d}")

            width_ar[i,j] = d
    for w in range(10):
        df[f"width_{w+1}"] = width_ar[:, w]
    
    batch_sizes = df["batch_size"].apply(parse_batch_size)

    df["batch_size_1"] = batch_sizes.apply(lambda x: x[0])
    df["batch_size_2"] = batch_sizes.apply(lambda x: x[1])

    df.drop(columns=["model_arch", "batch_size"], inplace=True)

    order_cols = df.columns.to_list()[0:5] + df.columns.to_list()[16:] + \
        df.columns.to_list()[5:16]
    df = df[order_cols]
    
    return df
    


def main():
    return None
    

if __name__ == "__main__":
    main()


