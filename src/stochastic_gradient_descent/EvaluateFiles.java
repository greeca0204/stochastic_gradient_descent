package stochastic_gradient_descent;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;

public class EvaluateFiles 
{
	public static void store_file_words_with_label( File directory, 
									    			Set<String> GLOBO_DICT,
									    			Table< ArrayList<String>, String, Integer > table) throws IOException 
	{
		String directory_label = null;
		
		int directory_read = 0;
		
		for (File file : directory.listFiles()) 
		{  
			
			ArrayList<String> document_words = new ArrayList<String>();
			
			String line; 
			BufferedReader br = new BufferedReader(new FileReader( file ));
			
			while((line = br.readLine()) != null) 
			{
				String[] words = line.split(" ");//those are your words
				
				String word;
				
				for (int i = 0; i < words.length; i++) 
				{
					word = words[i];
					
					document_words.add(word);
				}
			} 
			directory_label = file.getPath()
								  .toString()
								  .replaceAll("/[^/]*$", "")
								  .replaceAll("/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/practice_data/data/train/", "")
								  .replaceAll("/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/practice_data/data/test/", "")
								  .replaceAll("/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/data/train/", "")
								  .replaceAll("/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/data/test/", "");
			
			table.put( document_words, directory_label, directory_read);
			directory_read++;
		}   
		
	}
	
	public static void generate_word_frequency_count_struc(Set<String> GLOBO_DICT, 
														   Table< ArrayList<String>, String, Integer > fileDict,
														   Table< int[] , String , Integer > perceptron_input)
	{
		List<String> GLOBO_DICT_list = new ArrayList<>(GLOBO_DICT);
		
		//for (Map.Entry< ArrayList<String> , String > entry : fileDict.entrySet())
		for ( Cell< ArrayList<String>, String, Integer > cell: fileDict.cellSet() )
		{
			int[] cross_czech = new int[GLOBO_DICT_list.size()];
			
			for (String s : GLOBO_DICT_list)
			{
				for( String st : cell.getRowKey() ) 
				{
					if( st.equals(s) )
					{
						cross_czech[ GLOBO_DICT_list.indexOf( s ) ] = cross_czech[ GLOBO_DICT_list.indexOf( s ) ] +1;
					}
				}
			}
			perceptron_input.put( cross_czech , cell.getColumnKey(), cell.getValue() );	
		}
	}
	
	
}
