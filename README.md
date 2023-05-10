# Description 
The Stem takes as input an image I of size High x Width (28x28) and divides it into Np non-overlapping patches. Using Pytorch function nn.unfold(). Each patch has dimensions K x K (7x7). Each patch gets reshaped into a vector, then transformed to a feature vector of dimensions [32,16,49], Which will be input for backbone.
The backbone has collection of blocks, The MLP_BLOCK contains MLP1 and MLP2 with non-linear activation function and input of both MLP layer is transposed first. And another custom block is created with linear layers with nn.BatchNorm1d, nn.Dropout(0.25) and ReLU() – Activation Function. A series of Block B1,B2, …B5 is created. 
The classifier takes output of last Block B5, compute mean and then feed the mean features to SoftMax regression classifier to classify.
The model combines STEM, BACKBONE and CLASSIFIER into a network.
--
•	Stem module divided the input image into non-overlapping patches using PyTorch's nn.unfold() function. \
•	Backbone module consisted of MLP_BLOCK (with MLP1 and MLP2 with non-linear activation) and custom blocks (with linear layers, batch normalization, dropout, and ReLU activation). \
•	Classifier module computed the mean of the output from the last Block Bn and fed it to a SoftMax regression classifier for classification. \
•	Integrated STEM, BACKBONE and CLASSIFIER modules into a single network for KMNIST classification.
