"""     imports    """
import numpy as np
""" -------------- """

class perceptron( object ):
  """ 
     @parameter eta: float
     learnig rate (should be between 0.0 and 1.0)
     @parameter num_iter: integer
     Number of epochs

     @v_weight: 1D array
     Weights after training
     @v_error: list
     Number of false classification in evry epoch
  """

  def __init__(self, eta = 0.01, num_iter = 10):
    self.eta = eta
    self.num_iter = num_iter
    self.v_weight = []
    self.v_error = []

  def fit2data(self, x, y):
    """
       @parameter x: [n_samples, n_features], training data
       @parameter y: [n_samples], target values
    """
    self.v_weight = np.zeros( 1 + x.shape[1] )
    for sample in range( self.num_iter ):
      errs = 0
      for xi, target in zip( x, y ):
        deltaw = self.eta * ( target - self.predict( xi ) )
        self.v_weight[1:] += deltaw * xi
        self.v_weight[0] += deltaw
        errs += int( deltaw != 0.0 )
      self.v_error.append( errs )

    return ( self )

  def net_input(self, x):
    return ( np.dot( x, self.v_weight[1:] ) + self.v_weight[0] )

  def predict(self, x):
    return ( np.where( self.net_input( x ) >= 0.0, 1, -1 ) )

