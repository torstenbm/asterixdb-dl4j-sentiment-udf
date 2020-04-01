# AsterixDB User Defined Functions for Sentiment Analysis of Tweets by Recurrent Neural Networks 

UDF's for AsterixDB created during my master's thesis at the Norwegian University of Science and Technology. Will link to paper here once it is written.

The UDF is not working yet and still under development.

## Training Data 
The neural network in this model has been trained to process text by converting words to floating numbers and running these numbers through a compact embedding layer. To create these word-to-float conversions (or sentence-to-float-array conversions) I've used the words in the tweets in the Stanford Sentiment140 project, which can be downloaded [here](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip). To use the neural network for inference of new tweets one necesarily has to use the same word-to-float conversions as the model was trained on, therefore to run the UDF it is necesary to download the training data and update the path-variable inside of WordVec.java.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)