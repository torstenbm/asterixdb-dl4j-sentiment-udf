package dl4j;

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

public class CharVec {
    private HashMap<Character, Double> charToNumberMap = new HashMap<Character, Double>();

    // Replace with path to your own copy of the sentiment140 training data
    // http://help.sentiment140.com/for-students
    private static final String trainingCsvFile = "/Users/torsten/csv-processing/training.csv";

    public HashMap<Character, Double> initialize(){
        HashMap<Character, Double> charOccurences = new HashMap<Character, Double>();

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
                String tweetText = tweet[5].replaceAll("\"", "");

                for (char c : tweetText.toCharArray()) {
                    if (charOccurences.get(c) != null){
                        charOccurences.put(c, charOccurences.get(c) + 1);
                    } else {
                        charOccurences.put(c, 1.0);
                    }
                }

                tweetsRead = tweetsRead + 1;
            }  
        } catch (Exception e) {
            e.printStackTrace();
        } 

        charOccurences = sortByValue(charOccurences);
        double num = 1.0;
        for (Map.Entry<Character, Double> entry : charOccurences.entrySet()) {
            charToNumberMap.put(entry.getKey(), num);
            num++;
		}

        return charToNumberMap;
    }

    public double charToNumber(char c){
        if (charToNumberMap.get(c) != null){
            return charToNumberMap.get(c);
        }
        return 0.0;
    }
    public double[] stringToCharVector(String s){
        double[] vector = new double[280];
        for (int i = 0; i < s.length(); i++){
            vector[i] = charToNumber(s.charAt(i));
        }
        return vector;
    }
    public char[][] stringToCharMatrix(String s){
        char[][] matrix = new char[14][20];
        for (int i = 0; i < 20; i++){
            for (int j =  0;  j < 14;  j++){
                if (i*14 + j < s.length()){
                    matrix[j][i] = s.charAt(i*14 + j);
                }
            }
        }
        return matrix;
    }


    public static Reader getReader(String relativePath) throws UnsupportedEncodingException, FileNotFoundException {
        return new InputStreamReader(new FileInputStream(relativePath), "UTF-8");
    }

    public static HashMap<Character, Double> sortByValue(HashMap<Character, Double> hm) { 
        // Create a list from elements of HashMap 
        List<Map.Entry<Character, Double> > list = 
               new LinkedList<Map.Entry<Character, Double> >(hm.entrySet()); 
  
        // Sort the list 
        Collections.sort(list, new Comparator<Map.Entry<Character, Double> >() { 
            public int compare(Map.Entry<Character, Double> o1,  
                               Map.Entry<Character, Double> o2) 
            { 
                return (o2.getValue()).compareTo(o1.getValue()); 
            } 
        }); 
          
        // put data from sorted list to hashmap  
        HashMap<Character, Double> temp = new LinkedHashMap<Character, Double>(); 
        for (Map.Entry<Character, Double> aa : list) { 
            temp.put(aa.getKey(), aa.getValue()); 
        } 
        return temp; 
    } 
  

}
