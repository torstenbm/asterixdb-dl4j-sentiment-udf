/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.asterix.external.library;

import org.apache.asterix.external.api.IExternalScalarFunction;
import org.apache.asterix.external.api.IFunctionHelper;
import org.apache.asterix.external.library.java.base.JList;
import org.apache.asterix.external.library.java.base.JLong;
import org.apache.asterix.external.library.java.base.JBoolean;
import org.apache.asterix.external.library.java.base.JRecord;
import org.apache.asterix.external.library.java.base.JString;
import org.apache.asterix.external.library.java.base.JOrderedList;
import org.apache.asterix.external.library.dl4j.WordVec;
import org.apache.asterix.om.types.BuiltinType;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;

import java.io.File;


public class StreamedListsLSTMSentimentFunction implements IExternalScalarFunction {
    private int batchSize;
    private int batchPointer; 
    private int vectorLength;
    public MultiLayerNetwork net;
    public ParallelInference piModel;
    public WordVec customizedWordVec;
    public long startTime;
    public double[][] tweetVectorBatch;
    public JRecord[] outputRecords;


    @Override
    public void deinitialize() {}

    @Override
    public void evaluate(IFunctionHelper functionHelper) throws Exception {
        // Get input record/tweet
        System.out.println("Getting batch object");
        JRecord inputRecord = (JRecord) functionHelper.getArgument(0);

        // Extract and process text of tweet
        System.out.println("Extracting batch");
        JOrderedList tweetBatch = (JOrderedList) inputRecord.getValueByName("tweets");
        JLong tweetID = (JLong) inputRecord.getValueByName("id");

        // Build batch for processing
        System.out.println("Building batch for processing");
        for (int i = 0; i < tweetBatch.size(); i++){
            JRecord tweet = (JRecord) tweetBatch.getElement(i);
            JString tweetText = (JString) tweet.getValueByName("text");
            double[] tweetVector = customizedWordVec.sentenceToWordVec(tweetText.getValue(), vectorLength);
            tweetVectorBatch[i] = tweetVector;
        }
        System.out.println("Processing through RNN");
        // Convert tweetBatch to format understandable by DL4J neural networks
        INDArray features = Nd4j.create(tweetVectorBatch);

        // Run batch through the parallel inference model of the recurrent neural net
        INDArray networkOutput = piModel.output(features);

        // Sentiment probabilities will be probabilities at last iteration of recurrence 
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(vectorLength-1));

        // Convert to double array with an entry of 1 corresponding to positive and 0 to negative
        double[] predictedSentiments = probabilitiesAtLastWord.argMax(1).toDoubleVector();

        // Loop through batch of input records, add sentiment field and add to list 
        System.out.println("Matching up sentiments with tweets");
        for (int i = 0; i < batchSize; i++){
            String sentiment = predictedSentiments[i] == 0 ? "positive" : "negative";
            JRecord tweet = (JRecord) tweetBatch.getElement(i);
            tweet.setField("sentiment", new JString(sentiment));
            tweetBatch.add(outputRecords[i]);
        }

        // Get and populate output record
        System.out.println("Setting results");
        JRecord tweetBatchObject = (JRecord) functionHelper.getResultObject();
        tweetBatchObject.setField("id", tweetID);
        tweetBatchObject.setField("tweets", tweetBatch);

        // Set result object
        functionHelper.setResult(tweetBatchObject);

    }

    @Override
    public void initialize(IFunctionHelper functionHelper) throws Exception{
        // Number of words to allow in vector.
        vectorLength = 30;

        // Number of records to process at a time.
        batchSize = 50000;

        System.out.println("Started loading wordvectors");
        customizedWordVec = new WordVec();
        customizedWordVec.initialize();
        System.out.println("Wordvectors initialized");

        System.out.println("Initialization of Neural Net started");
        File f = new File("/lhome/torstebm/asterixdb-dl4j-sentiment-udf/src/main/java/org/apache/asterix/external/library/dl4j/1m_rnn_customizedWordVec_model.zip");
        boolean saveUpdater = false;    
        net = MultiLayerNetwork.load(f, saveUpdater);
        System.out.println("Neural Net Initialized");

        // TODO: Load ParallelInference model directly to shorten initialize function
        System.out.println("Initializing Parallel Inference");
        piModel = new ParallelInference.Builder(net)
            .inferenceMode(InferenceMode.BATCHED)
            // max size of batch for BATCHED mode. Set with respect to your environment (i.e. gpu memory)
            .batchLimit(15000)
            // set this value to number of available computational devices
            .workers(2)
            .build();
        System.out.println("Parallel inference initialized");

        System.out.println("I am using this file, right?");

        // Initialize batching data structures
        tweetVectorBatch = new double[batchSize][vectorLength];
        outputRecords = new JRecord[batchSize];
        batchPointer = 0;

        startTime = System.nanoTime();
        System.out.println("End of initialize");
    }
}