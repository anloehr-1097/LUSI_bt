import tensorflow as tf
import numpy as np
import functools
import matplotlib.pyplot as plt
from primefac import primefac
from pprint import pprint
import pandas as pd

class Periphery:
    def __init__(self, train_data=None, test_data=None, phi=np.array([]), batch_size_1=32, batch_size_2=32) -> None:
        
        """Bind all data and objetcs for Lusi method.
        
        Parameters:
        train_data :: tuple of np.ndarrays - (x_train, y_train) | None
            Training data.
        
        test_data :: tuple of np.ndarrays - (x_test, y_test) | None
            Test data.
            
        phi :: np.ndarray[list[function]]
            A numpy array of list of functions to be applied on x_data.
        
        batch_size_1 :: int
            Batch size B from paper. This batch size is used in training.
            Default value: 32.
            
        batch_size_2 :: int
            Batch size B' from paper. This batch size is used in training.
            Default value: 32.        
        """
        
        # create attributes with None values, will be populated in set_phi method
        self.phi = np.array([])
        self.phi_eval_train = tf.constant([])
        self.phi_eval_test = tf.constant([])
        self.train_data = None
        self.test_data = None
        
        self.set_phi(phi)
        self.set_train_data(train_data)
        self.set_test_data(test_data)

        return None
    

    def generate_batch_data(self, batch_size_1=64, batch_size_2=64, train_only=False):
        """Generate batch datasets for training and test data.
        
        
        Parameters:
        
        batch_size_1 :: int
            See paper for meaning of batch_size_1 denoted as B.
        
        batch_size_2 :: int
            See paper for meaning of batch_size_1 denoted as B'.
            
        train_only :: Bool
            Create batch dataset for train data only.
        
        Returns:
        A tuple of batched datasets.
        """
        
        # Check for provision of relevant data.
        if not self.train_data:
            raise ValueError("No training data provided.")
            
        if not (self.test_data or train_only):
            raise ValueError("No test data provided.")
            
        if not (tf.size(self.phi_eval_train) > 0).numpy():
            raise ValueError("No phi evaluation data for training data provided.")
        
        #
        dataset_train = tf.data.Dataset.from_tensor_slices((self.phi_eval_train,
                                                      self.train_data[0], self.train_data[1]))   
        
        batch_dataset_1_train = dataset_train.shuffle(buffer_size=1024).batch(batch_size_1,
                                                                              drop_remainder=True)
        batch_dataset_2_train = dataset_train.shuffle(buffer_size=1024).batch(batch_size_2,
                                                                              drop_remainder=True)

        batch_dataset_train = tf.data.Dataset.zip((batch_dataset_1_train, batch_dataset_2_train))   
        
        
        if train_only:
            return (batch_dataset_train, )
        
        
        # also generate batch dataset for test data
        if not (tf.size(self.phi_eval_test) > 0).numpy():
            raise ValueError("No phi evaluation data for test data provided.")
        
        dataset_test = tf.data.Dataset.from_tensor_slices((self.phi_eval_test,
                                                            self.test_data[0], self.test_data[1]))   

        batch_dataset_1_test = dataset_test.shuffle(buffer_size=1024).batch(batch_size_1,
                                                                              drop_remainder=True)
        batch_dataset_2_test = dataset_test.shuffle(buffer_size=1024).batch(batch_size_2,
                                                                              drop_remainder=True)

        batch_dataset_test = tf.data.Dataset.zip((batch_dataset_1_test, batch_dataset_2_test)) 
            
        return (batch_dataset_train, batch_dataset_test)
   
            
    def apply_phi(self, which_dataset="both") -> None:
        """ Evaluate predicates on data and store values.

        Parameters: 
        which_dataset :: str
            One of "train", "test", "both".
            Indicates which dataset phi should be applied to.

        """

        if which_dataset == "both":
            if not (self.train_data and self.test_data):
                raise ValueError("Some data missing. Check if both train and test data provided.")

            train_x = self.train_data[0]
            test_x = self.test_data[0]

            phi_x_train = np.asarray([self.phi[i](train_x[j]) for j in range(train_x.shape[0]) for i in range(self.phi.shape[0])])
            phi_x_test = np.asarray([self.phi[i](test_x[j]) for j in range(test_x.shape[0]) for i in range(self.phi.shape[0])])

            phi_x_train = phi_x_train.reshape(train_x.shape[0], self.phi.shape[0])
            phi_x_test = phi_x_test.reshape(test_x.shape[0], self.phi.shape[0])

            phi_x_train = tf.convert_to_tensor(phi_x_train, dtype=tf.float32)
            phi_x_test = tf.convert_to_tensor(phi_x_test, dtype=tf.float32)

            self.phi_eval_train = phi_x_train
            self.phi_eval_test = phi_x_test

        elif which_dataset == "train":

            train_x = self.train_data[0]
            phi_x_train = np.asarray([self.phi[i](train_x[j]) for j in range(train_x.shape[0]) for i in range(self.phi.shape[0])])
            phi_x_train = phi_x_train.reshape(train_x.shape[0], self.phi.shape[0])
            phi_x_train = tf.convert_to_tensor(phi_x_train, dtype=tf.float32)
            self.phi_eval_train = phi_x_train

        else:
            test_x = self.test_data[0]
            phi_x_test = np.asarray([self.phi[i](test_x[j]) for j in range(test_x.shape[0]) for i in range(self.phi.shape[0])])
            phi_x_test = phi_x_test.reshape(test_x.shape[0], self.phi.shape[0])
            phi_x_test = tf.convert_to_tensor(phi_x_test, dtype=tf.float32)
            self.phi_eval_test = phi_x_test
            
        return None
    
        
    def set_phi(self, phi : list) -> None:
        if not phi.size > 0:
            return None
        
        self.phi = phi
        
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
        if not train_data:
            return None
            
        
        self.train_data = train_data
        self.train_data_x = train_data[0]
        self.train_data_y = train_data[1]
        
        if self.phi.size > 0:
            self.apply_phi("train")

        return None
    
    
    def set_test_data(self, test_data):
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
        size_of_excerpt = np.floor(size_of_excerpt * data[0].shape[0]).astype(int)
    
    if not size_of_excerpt <= data[0].shape[0]:
        raise IndexError("Sample Size too large.")
    
    permute = np.random.permutation(np.arange(size_of_excerpt))

    if balanced:
        y_cls = set(data[1])
        print(f"No. of classes = {len(y_cls)}")
        if len(y_cls) != 2:
            raise Exception("Pass data for binary classification problem.")
        
        y_1_cls = y_cls.pop()
        y_2_cls = y_cls.pop()
        
        x_1 = data[0][data[1] == y_1_cls]
        x_2 = data[0][data[1] == y_2_cls]

        y_1 = data[1][data[1] == y_1_cls]
        y_2 = data[1][data[1] == y_2_cls]
        ind_1 = np.random.randint(low=0, high=x_1.shape[0],
                                          size=(size_of_excerpt//2).astype(int))
        ind_2 = np.random.randint(low=0, high=x_2.shape[0],
                                          size=(size_of_excerpt - \
                                          (size_of_excerpt//2))).astype(int)
        
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

    # determine prime factors of n to calc. no. of rows and cols
    pfs = list(primefac(n))
    three_quart = np.floor(len(pfs) * 0.75).astype(int)
    rowcount = np.prod(pfs[: three_quart])
    colcount = np.prod(pfs[three_quart : ])

    # select n datapoints from data, bring into grid shape for easy handling
    random_indices = np.random.randint(0, high=data[0].shape[0], size=n)
    random_indices = np.reshape(random_indices, (rowcount, colcount))
    
    # determine size of figure dynamically
    size = np.max([colcount, rowcount]) * 3
    fig, ax = plt.subplots(nrows=rowcount, ncols=colcount, figsize=(size, size))
    
    # populate grid with images + labels
    for i in range(rowcount):
        for j in range(colcount):
            ax[i,j].imshow(data[0][random_indices[i,j]])
            ax[i,j].set_title("pred: ({:.2f} -> {:.0f}),  true: {:.0f}".format( \
                data[2][random_indices[i,j]][0],
                np.round(data[2][random_indices[i,j]][0]),
                data[1][random_indices[i,j]]))
            # ax[i,j].set_title(f"pred: {data[2][random_indices[i,j]][0]}, true: {data[1][random_indices[i,j]]}")
            ax[i,j].axis("off")
    
    return None




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


def run_config(conf, no_of_runs=10):
    """Run a job/ config and return results.
    
    Interpret config dictionary, build model, run training on data and
    evaluate model on test dataset. Do this no_of_runs times.

    Parameters:
    no_of_runs :: int
        Number of runs to make for same model to gauge variance of results.

    Returns:
    list of results for given metric.
    
    """

    return None



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
    "no_of_predicates" : [3, 6, 8, 11],
    "epochs_training" : [1, 2, 5, 10, 15, 20, 30]}


config_dict = {hparam : np.asarray(v) for hparam, v in config_dict.items()}


def main():
    pprint(random_experiment(config_dict))
    return None
    

if __name__ == "__main__":
    main()


