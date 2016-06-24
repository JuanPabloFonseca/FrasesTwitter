package itam.twitter.base;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;

public class Rule {
	private HashMap<String,Double> words;
	private String name;
	private LinkedList<String > sample;
	private int maxSample;
	public int count;
	public  Rule() {
		words = new HashMap<>();
		sample = new LinkedList<>();
		maxSample = 10;
	}
	public  Rule(int maxi) {
		words = new HashMap<>();
		sample = new LinkedList<>();
		maxSample = maxi;
	}
	
	public void setWord(String word,double value){
		words.put(word, value);
	}
	public void setName(String nombre){
		name = nombre;
	}
	public String getName(){
		return name;
	}
	public double returnRanking(String tweet){
		String ar [] = tweet.split(" ");
		double rank =0;
		for (String w:ar){
			rank += words.containsKey(w)?words.get(w):0; 
		}
		return rank;
	}
	
	public String toString(){
		String ruleS="Regla: " +name;
		Iterator<String> iterator = words.keySet().iterator();
		  
		while (iterator.hasNext()) {
		   String key = iterator.next().toString();
		   String value = words.get(key).toString();
		  
		   ruleS+= key + " " + value+" | ";
		}
		return ruleS;
	}
	
	public void addToSample(String newSample){
		if(sample.size() <=maxSample){
			sample.add(newSample);
		}
	}
	
	public void printSamples(){
		printSamples(maxSample);
	}
	public void printSamples(int n){
		System.out.println("Ejemplos de "+name +" ");
		int i=1;
		for(String s : sample){
			System.out.println(i+"\t"+s);
			i++;
		}
	}
	
	public static Rule parseRule(String name, String words[],Double value[]){
		Rule r=null;
		
		r = new Rule();
		r.setName(name);
		for (int i=0; i<words.length;i++){
			r.setWord(words[i], value[i]);
		}
		return r;
	}
}
