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


public class StreamedRecordsLSTMSentimentFunction implements IExternalScalarFunction {
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
        JRecord inputRecord = (JRecord) functionHelper.getArgument(0);

        // Extract and process text of tweet
        JString tweetText = (JString) inputRecord.getValueByName("text");
        JLong tweetID = (JLong) inputRecord.getValueByName("id");
        double[] tweetVector = customizedWordVec.sentenceToWordVec(tweetText.getValue(), vectorLength);

        // Add record to list
        JRecord outputRecord = (JRecord) functionHelper.getResultObject();
        outputRecord.setField("text", tweetText);
        outputRecord.setField("id", tweetID);

        // Build batch of vectors to be processed by RNN, while keeping track of
        // order for when we must match RNN sentiment output with record after processing
        outputRecords[batchPointer] = inputRecord;
        tweetVectorBatch[batchPointer] = tweetVector;
        batchPointer++;

        // If batch is full
        if (batchPointer >= batchSize){
            // Convert tweetBatch to format understandable by DL4J neural networks
            INDArray features = Nd4j.create(tweetVectorBatch);

            // Run batch through the parallel inference model of the recurrent neural net
            INDArray networkOutput = piModel.output(features);

            // Sentiment probabilities will be probabilities at last iteration of recurrence 
            INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(vectorLength-1));

            // Convert to double array with an entry of 1 corresponding to positive and 0 to negative
            double[] predictedSentiments = probabilitiesAtLastWord.argMax(1).toDoubleVector();
        
            // Init list of processed tweets that will be a field of the return record,
            // accessed through TweetBatch.tweets through SQL++ in the AsterixDB query interface
            JOrderedList tweetBatch = new JOrderedList(BuiltinType.ANY);

            // Loop through batch of input records, add sentiment field and add to list 
            for (int i = 0; i < batchSize; i++){
                String sentiment = predictedSentiments[i] == 0 ? "positive" : "negative";
                outputRecords[i].setField("sentiment", new JString(sentiment));
                tweetBatch.add(outputRecords[i]);
            }

            // Get and populate output record
            JRecord tweetBatchObject = (JRecord) functionHelper.getResultObject();
            tweetBatchObject.setField("id", tweetID);
            tweetBatchObject.setField("isMaster", new JBoolean(true));
            tweetBatchObject.setField("tweets", tweetBatch);

            // Set result object
            functionHelper.setResult(tweetBatchObject);
 
            // Reset batch pointer as we prepare next batch
            batchPointer = 0;

            // Track batch processing time
            long totalTime = (System.nanoTime() - startTime);
            System.out.println("Batch time: " + String.valueOf(totalTime) + " nanoseconds");
            startTime = System.nanoTime(); 

        } else {
            // Return dummy record due to UDF contract of every input needing an output
            JRecord tweetBatchObject = (JRecord) functionHelper.getResultObject();
            JOrderedList tweetBatch = new JOrderedList(BuiltinType.ANY);
            tweetBatchObject.setField("id", tweetID);
            tweetBatchObject.setField("tweets", tweetBatch);
            tweetBatchObject.setField("isMaster", new JBoolean(false));
            functionHelper.setResult(tweetBatchObject);
        }
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


        // Initialize batching data structures
        tweetVectorBatch = new double[batchSize][vectorLength];
        outputRecords = new JRecord[batchSize];
        batchPointer = 0;

        startTime = System.nanoTime();
    }
}