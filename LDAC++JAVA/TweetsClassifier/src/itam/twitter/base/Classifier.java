package itam.twitter.base;
import java.io.*;
import java.util.Arrays;
import java.util.LinkedList;
public class Classifier {
	private LinkedList<Rule> topics; 
	
	public Classifier(){
		topics = new LinkedList<Rule>();
	}
	public void printRules(){
		System.out.println("Imprimiendo Reglas");
		for(Rule r: topics){
			System.out.println(r.toString());
		}
	}
	
	public boolean readRules(String fileName){
		String sCurrentLine;
		Rule r;
		String rName,words[];
		Double values[];
		BufferedReader br=null;
		System.out.println("Leyendo reglas... ");
		try{
			br = new BufferedReader(new FileReader(fileName));
			
			words = new String[0];
			values = new Double[0];
			rName="";
			sCurrentLine = br.readLine();
			if(sCurrentLine.toLowerCase().contains("topic")){
				rName = "topic_" + sCurrentLine.substring(sCurrentLine.indexOf("topic")+6).trim();
			}
			System.out.println("Evaluando regla: "+rName);
			int i;
			while ((sCurrentLine = br.readLine()) != null) {				
				if(sCurrentLine.toLowerCase().contains("topic")){
					if(words.length==0){ System.out.println("No hubo palabras para la regla "+rName);return false;}
					r = Rule.parseRule(rName, words,values);
					topics.add(r);
					rName = "topic_" + sCurrentLine.substring(sCurrentLine.indexOf("topic")+6).trim();
					System.out.println("Evaluando regla: "+rName);
					words = new String[0];
					values = new Double[0];
				}
				else{
					i = words.length;
					words = Arrays.copyOf(words,i+1);
					values = Arrays.copyOf(values,i+1);
					String ar[] = sCurrentLine.trim().split("\\s++");
					words[i] = ar[0];
					values[i] = Double.parseDouble(ar[1]);
				}
				
			}
			if(words.length>0){ //LAST RULE
				r = Rule.parseRule(rName, words,values);
				topics.add(r);
			}
			
			System.out.println("Reglas Listas... \n\n");
			return true;
		}catch (IOException e){
			System.out.println("Fallo algo en el archivo de reglas...");
			e.printStackTrace();
			return false;
		}finally{
			try {
				if (br != null)br.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	public void readAndClassifyTweets(String fileName){
		BufferedReader br=null;
		String sCurrentLine;
		double max;
		Rule rMax;
		int tweets,countOthers=0;
		
		try{
			br = new BufferedReader(new FileReader(fileName));
			tweets=0;
			while ((sCurrentLine = br.readLine()) != null) {
				//MAYBE CLEAN THE TWEET???
				max= -1;
				rMax = null;
				for(Rule currentTopic:topics) {
					double res = currentTopic.returnRanking(sCurrentLine);
					if(res > max){
						max = res;
						rMax = currentTopic;
					}
				}
				if(rMax != null){ //check this
					rMax.count++;
					rMax.addToSample(sCurrentLine);
				}else{
					countOthers ++;
				}
				tweets++;
			}
			System.out.println("Reporte: ");
			System.out.println("=================================================================================");
			System.out.println("De los "+topics.size()+" topicos, se analizaron "+tweets +" tweets: ");
			for(Rule ts : topics){
				System.out.println("Del topico  "+ts.getName()+" pertenecen "+ts.count+" tweets");	
			}
			System.out.println("\n\n");
			for(Rule ts : topics){
				System.out.println("=================================================================================");
				ts.printSamples();
			}
			
			
		}catch (IOException e){
			System.out.println("Fallo algo en archivo de tweets...");
			e.printStackTrace();
		}finally{
			try {
				if (br != null)br.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	public static void main(String[] args) {
		if(args.length<2) {System.out.println("uso Classifier <reglas> <nombreArchivo>");return;}
		Classifier c = new Classifier();
		if(c.readRules(args[0])){
			c.readAndClassifyTweets(args[1]);
			//	System.out.println("TODO: Calificar archivo: " + args[1]);
		}
		//c.printRules();
		
		
	}

}
