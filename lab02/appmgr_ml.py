""" use the percepton class to classify data          """
""" first we need to fetch the data and fit the model """
""" ------------------------------------------------- """

import pandas as pd
import numpy as np
from perceptron import perceptron as Neuron
import matplotlib.pyplot as plot

def connect2db():
  address = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
  db_name = 'iris/iris.data'
  connecting_string = address + db_name
  database = pd.read_csv( connecting_string, header = None )
  if not database.empty:
    print ( ' --> Success! Data set has been fetched from the web page' )
  else:
    print ( ' --> Failure! It seems there is no data!' )
    exit( 1 )
  return ( database )

def print2screen( db, rows = 10 ):
  datah = db.head( rows )
  datat = db.tail( rows )
  print ( datah )
  print ( datat )

def extract_features( db ):
  """ iteger location """
  y_raw = db.iloc[0:100, 4].values
  # --> prepare labels
  y = np.where(y_raw == 'Iris-setosa', -1, 1)
  # --> prepare samples
  x = db.iloc[0:100, [0,2]].values
  if len(x)!= 0 and len(y) != 0:
    return ( x, y )
  else:
    print ( ' --> problem with feature extraction ' )
    exit ( 1 )

def plot_raw_data( x_in, y_in ):
  plot.figure( 1 )
  plot.scatter( x_in[:50, 0], x_in[:50, 1], color = 'red', marker = 'o', label = 'setosa')
  plot.scatter( x_in[50:100, 0], x_in[50:100, 1], color = 'black', marker = 'x', label = 'versicolor')
  plot.xlabel( ' petal lenght [cm] ' )
  plot.ylabel( ' sepal length [cm] ' )
  plot.legend( loc = 'upper left' )
  plot.show(block = False)
  return ( )

def plot_decision_boundary( x_in, y_in, classifier, reso = 0.01 ):
  from matplotlib.colors import ListedColormap
  markers = ( 's', 'x', 'o', '_', 'v' )
  colors = ( 'red', 'blue', 'gray', 'yellow', 'cyan' )
  col_map = ListedColormap( colors[:len( np.unique( y_in ) ) ] )
  # --> plot decision hyper-plane
  x1_min, x1_max = x_in[:, 0].min() - 1, x_in[:, 0].max() + 1
  x2_min, x2_max = x_in[:, 1].min() - 1, x_in[:, 1].max() + 1
  x_x1, x_x2 = np.meshgrid( np.arange( x1_min, x1_max, reso ), np.arange( x2_min, x2_max, reso ) )
  out_data = classifier.predict( np.array( [x_x1.ravel(), x_x2.ravel() ] ).T )
  out_data = out_data.reshape( x_x1.shape )
  plot.figure( 3 )
  plot.contourf( x_x1, x_x2, out_data, alpha = 0.4, cmap = col_map )
  plot.xlim( x_x1.min(), x_x1.max() )
  plot.ylim( x_x2.min(), x_x2.max() )
  for index, cl in enumerate( np.unique( y_in ) ):
    plot.scatter( x = x_in[y == cl, 0 ], y = x_in[y == cl, 1], alpha = 0.8, c = col_map( index ), marker = markers[ index ], label = cl )

if __name__ == "__main__":
  DB = connect2db()
  print2screen( DB )
  x, y = extract_features( DB )
  plot_raw_data( x, y )
  # --> create and train the perceptron
  print ( ' --> Training ' )
  n = Neuron( eta = 0.05, num_iter = 10 )
  n.fit2data( x, y )
  # --> summary and results
  print ( ' --> Results' )
  plot.figure( 2 )
  plot.plot( range( 1, len( n.v_error ) + 1 ), n.v_error, marker = 'o' )
  plot.xlabel( ' Epochs ' )
  plot.ylabel( ' Number of errors ' )
  plot.show(block = False)
  # --> now use the predict to plot the decision boundary
  plot_decision_boundary( x, y, classifier = n, reso = 0.05 )
  plot.xlabel( ' sepal length [cm] ' )
  plot.ylabel( ' petal lenght [cm] ' )
  plot.legend( loc = 'upper left' )
  plot.show()


