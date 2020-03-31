package org.apache.asterix.external.library.dl4j;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.UnsupportedEncodingException;
import java.io.Reader;
import java.io.InputStreamReader;
import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;

public class WordVec {
    private HashMap<String, Double> wordToNumber = new HashMap<String, Double>();
    
    // Replace with path to your own copy of the sentiment140 training data
    // http://help.sentiment140.com/for-students
    private static final String trainingCsvFile = "/lhome/torstebm/deeplearning4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/training.csv";
    
    public int getDictSize(){
        return this.wordToNumber.size();
    }
    public HashMap<String, Double> initialize(){
        HashMap<String, Double> wordOccurences = new HashMap<String, Double>();

        try {
            CsvParserSettings settings = new CsvParserSettings();
            CsvParser parser = new CsvParser(settings);
            Reader reader = getReader(trainingCsvFile);
            parser.beginParsing(reader);

            String[] tweet;
            int tweetsRead = 0;
            while ((tweet = parser.parseNext()) != null) {
                if (tweetsRead >= 1600000) {
                    break;
                }
                String tweetText = tweet[1].replaceAll("\"", "");
                
                String[] words = tweetText.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");

                for (String s : words) {
                    if (wordOccurences.get(s) != null){
                        wordOccurences.put(s, wordOccurences.get(s) + 1);
                    } else {
                        wordOccurences.put(s, 1.0);
                    }
                }

                tweetsRead = tweetsRead + 1;
            }  
        } catch (Exception e) {
            e.printStackTrace();
        } 

        wordOccurences = sortByValue(wordOccurences);
        double num = 1.0;
        for (Map.Entry<String, Double> entry : wordOccurences.entrySet()) {
            wordToNumber.put(entry.getKey(), num);
            num++;
		}

        return wordToNumber;
    }

    public double wordToNumber(String c){
        if (wordToNumber.get(c) != null){
            return wordToNumber.get(c);
        }
        return 0.0;
    }

    public double[] sentenceToWordVec(String s, int vectorLength){
        double[] vector = new double[vectorLength];
        String[] words = s.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");

        for (int i = 0; i < vectorLength; i++){
            if (i < words.length){
                vector[i] = wordToNumber(words[i]);
            } else {
                vector[i] = 0.0;
            }
        }
        return vector;
    }


    public static Reader getReader(String relativePath) throws UnsupportedEncodingException, FileNotFoundException {
        return new InputStreamReader(new FileInputStream(relativePath), "UTF-8");
    }



    public static HashMap<String, Double> sortByValue(HashMap<String, Double> hm) { 
        // Create a list from elements of HashMap 
        List<Map.Entry<String, Double> > list = 
               new LinkedList<Map.Entry<String, Double> >(hm.entrySet()); 
  
        // Sort the list 
        Collections.sort(list, new Comparator<Map.Entry<String, Double> >() { 
            public int compare(Map.Entry<String, Double> o1,  
                               Map.Entry<String, Double> o2) 
            { 
                return (o2.getValue()).compareTo(o1.getValue()); 
            } 
        }); 
          
        // put data from sorted list to hashmap  
        HashMap<String, Double> temp = new LinkedHashMap<String, Double>(); 
        for (Map.Entry<String, Double> aa : list) { 
            temp.put(aa.getKey(), aa.getValue()); 
        } 
        return temp; 
    } 
}
