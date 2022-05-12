import tensorflow as tf
import numpy as np
import functools

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
        
        self.phi_eval_train = None
        self.phi_eval_test = None
        self.train_data = None
        self.test_data = None
        
        self.set_phi(phi)
        
        self.set_train_data(train_data)
        self.set_test_data(test_data)
        
        
        """
        if self.train_data:
            self.generate_batched_data(train=True)
        
        if self.test_data:
            self.generate_batched_data(train=False)
        """  
    
    def generate_batch_data(self, batch_size_1=64, batch_size_2=64, train_only=False):
        """Generate batch datasets for training and test data.
        
        
        Parameters:
        
        batch_size_1 :: int
            
        batch_size_2 :: int
            
        train_only :: Bool
        
        Returns:
        A tuple of batched datasets.
        
        """
        
        # Check for provision of relevant data.
        if not self.train_data:
            raise ValueError("No training data provided.")
            
        if not (self.test_data or train_only):
            raise ValueError("No test data provided.")
            
        if not self.phi_eval_train:
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
        if not self.phi_eval_test:
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