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
import org.apache.asterix.external.api.IJObject;
import org.apache.asterix.external.library.java.base.JOrderedList;
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
import java.util.List;
import java.util.ArrayList;


public class StoredDataLSTMSentimentFunction implements IExternalScalarFunction {
    private int batchSize;
    private int vectorLength;
    public MultiLayerNetwork net;
    public ParallelInference piModel;
    public WordVec customizedWordVec;
    public long startTime;

    @Override
    public void deinitialize() {}

    @Override
    public void evaluate(IFunctionHelper functionHelper) throws Exception {
        startTime = System.nanoTime();

        // Read input records
        JOrderedList inputRecordsOrdered = (JOrderedList) functionHelper.getArgument(0);
        List<IJObject> inputRecords = inputRecordsOrdered.getValue();
        int numRecords = inputRecords.size();

        // Initialize batch
        // TODO: Consider dividing batch up into smaller batches for large numbers of records
        double[][] tweetVectorBatch = new double[numRecords][vectorLength];

        // Convert tweets to vectors and put in batch
        for (int i = 0; i < numRecords; i++){
            JRecord tweetRecord = (JRecord) inputRecords.get(i);
            JString tweetText = (JString) tweetRecord.getValueByName("text");
            

            tweetVectorBatch[i] = customizedWordVec.sentenceToWordVec(tweetText.getValue(), vectorLength);
        }
        // Convert tweetBatch to format understandable by DL4J neural networks
        INDArray features = Nd4j.create(tweetVectorBatch);

        // Run batch through the parallel inference model of the recurrent neural net
        INDArray networkOutput = piModel.output(features);

        // Sentiment probabilities will be probabilities at last iteration of recurrence 
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(vectorLength-1));

        // Convert to double array with an entry of 1 corresponding to positive and 0 to negative
        double[] predictedSentiments = probabilitiesAtLastWord.argMax(1).toDoubleVector();

        // Populate output list with enriched tweets
        List<IJObject> outputRecords = new ArrayList<IJObject>();

        int unprocessedRecords = numRecords;
        int currentBatch = 0;
        while (unprocessedRecords > 0){
            if (unprocessedRecords > batchSize) {
                currentBatch = currentBatch + batchSize;
            } else {
                currentBatch = currentBatch + unprocessedRecords;
            }
            for (int i = 0; i < currentBatch; i++){
                JRecord tweetRecord = (JRecord) inputRecords.get(i);
                String sentiment = predictedSentiments[i] == 0 ? "positive" : "negative";
                tweetRecord.setField("sentiment", new JString(sentiment));
                outputRecords.add(tweetRecord);
            }
            unprocessedRecords = unprocessedRecords - batchSize;
        }
        
        // Tracking processing time
        long totalTime = (System.nanoTime() - startTime);
        System.out.println("Total classification time: " + String.valueOf(totalTime) + " nanoseconds");
        
        // Set result
        JOrderedList outputRecordsOrdered = (JOrderedList) functionHelper.getResultObject();
        functionHelper.setResult(outputRecordsOrdered);
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
            .workers(2)
            .build();
        System.out.println("Parallel inference initialized");
    }
}