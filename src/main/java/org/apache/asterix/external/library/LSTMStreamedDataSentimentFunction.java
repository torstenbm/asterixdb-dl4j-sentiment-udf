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
import org.apache.asterix.external.library.java.base.JRecord;
import org.apache.asterix.external.library.java.base.JString;
import org.apache.asterix.external.library.dl4j.WordVec;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;

import java.io.File;


public class LSTMStreamedDataSentimentFunction implements IExternalScalarFunction {
    private int batchSize;
    private int batchPointer; 
    private int vectorLength;
    public MultiLayerNetwork net;
    public ParallelInference piModel;
    public WordVec customizedWordVec;
    public long startTime;
    public double[][] tweetVectorBatch;


    @Override
    public void deinitialize() {}

    @Override
    public void evaluate(IFunctionHelper functionHelper) throws Exception {
        // startTime = System.nanoTime();

        // Read input records
        JList inputRecord = (JRecord) functionHelper.getArgument(0);


        // Extract and process text of tweet
        JString tweetText = (JString) inputRecord.getValueByName("text");
        JLong tweetID = (JLong) inputRecord.getValueByName("id");
        double[] tweetVector = customizedWordVec.sentenceToWordVec(tweetText.getValue(), vectorLength);

        // Add corresponding output record to list
        JRecord outputRecord = (JRecord) functionHelper.getResultObject();
        outputRecord.setField("text", tweetText);
        outputRecord.setField("id", tweetID);

        outputRecords[batchPointer] = outputRecord;
        batchPointer++;

        // Put record in batch and keep track of output order
        tweetVectorBatch[batchPointer] = tweetVector;

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
        
            // Return results to AsterixDB
            for (int i = 0; i < batchSize; i++){
                String sentiment = predictedSentiments[i] == 0 ? "positive" : "negative";
                outputRecords[i].setField("sentiment", new JString(sentiment));
                
                functionHelper.setResult(outputRecord);
            }

            // Reset batch pointer
            batchPointer = 0;
        }

        // Else, do nothing
        
    }

    @Override
    public void initialize(IFunctionHelper functionHelper) throws Exception{
        // Number of words to allow in vector.
        vectorLength = 30;

        // Number of records to process at a time.
        batchSize = 100000;

        //https://deeplearning4j.org/workspaces
        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        System.out.println("Started loading wordvectors");
        customizedWordVec = new WordVec();

        
        customizedWordVec.initialize();
        System.out.println("Wordvectors initialized");

        System.out.println("Initialization of Neural Net started");
        File f = new File("/lhome/torstebm/asterixdb-dl4j-sentiment-udf/src/main/java/org/apache/asterix/external/library/dl4j/1m_rnn_customizedWordVec_model.zip");
        boolean saveUpdater = false;
        net = MultiLayerNetwork.load(f, saveUpdater);
        System.out.println("Neural Net Initialized");

        // TODO: Load ParallelInference model directly
        System.out.println("Initializing Parallel Inference");
        piModel = new ParallelInference.Builder(net)
            // BATCHED mode is kind of optimization: if number of incoming requests is too high - PI will be batching individual queries into single batch. If number of requests will be low - queries will be processed without batching
            .inferenceMode(InferenceMode.BATCHED)
            // max size of batch for BATCHED mode. you should set this value with respect to your environment (i.e. gpu memory amounts)
            .batchLimit(15000)
            // set this value to number of available computational devices, either CPUs or GPUs
            .workers(2)s
            .build();
        System.out.println("Parallel inference initialized");


        // Initialize batch 
        double[][] tweetVectorBatch = new double[batchSize][vectorLength];
        batchPointer = 0;
        JRecord[] outputRecords = new JRecord[batchSize];
    }
}