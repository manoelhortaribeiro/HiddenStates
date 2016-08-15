========================================================================================================================

** This folder contains the data-sets used for the experiments done:

------------------------------------------------------------------------------------------------------------------------

* The NATOPS dataset
"This dataset contains three pairs of body-hand gestures used when handling aircraft on the deck of an aircraft carrier.
 The observation features include automatically tracked 3D body postures and hand shapes. The body feature includes 3D
 joint velocities for left/right elbows and wrists, and represented as a 12D feature vector. The hand feature includes
 probability estimates of five predefined hand shapes - opened/closed palm, thumb up/down, and "no hand". The fifth
 shape, no hand, was dropped in the final representation, resulting in an 8D feature vector. The dataset was sampled at
 20 FPS."

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

* Thhe ArmGesture dataset
"This dataset includes the six arm gestures. Observation features include automatically tracked 2D joint angles and 3D
euclidean coordinates for left/right shoulders and elbows; each observation is represented as a 20D feature vector. The
dataset was collected from 13 participants with an average of 120 samples per class (exact sample counts per class are
[88, 117, 118, 132, 179, 90])."

------------------------------------------------------------------------------------------------------------------------

* Both the original datasets  datasets were obtained from Yale Song, and can be found here:

- http://people.csail.mit.edu/yalesong/cvpr12/

------------------------------------------------------------------------------------------------------------------------

* Inside each of the dataset's folder, the AG.mat and the NT.mat files correspond to the original data-sets. The other
files in the format:

- X_Y{c,d}.mat

* Contain the binarized datasets, where all the labels in X became a new label 0 and all the labels in Y, 1.

========================================================================================================================
