package stochastic_gradient_descent;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

public class Main 
{
	//static String PATH = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/data/train";
	static String PATH = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/practice_data/data/train";
	
	final static Table< int[] , String , Integer > train_freq_count_against_globo_dict = HashBasedTable.create();
	final static Table< int[] , String , Integer > test_freq_count_against_globo_dict = HashBasedTable.create();
	
	final static Table< ArrayList<String>, String, Integer > train__list_of_file_words = HashBasedTable.create();
	final static Table< ArrayList<String>, String, Integer > test__list_of_file_words = HashBasedTable.create();
	
	
	//all of the words in all of the files
	static Set<String> GLOBO_DICT = new HashSet<String>();
	
	public static void main(String[] args) throws IOException 
	{		
		
		
		//each of the diferent categories
		//String[] categories = { "/atheism", "/politics", "/science", "/sports"};
		String[] categories = { "/atheism", "/sports"};
		//String[] categories = { "/politics", "/science"};
		
		//cycle through all categories once to populate the global dict
		for(int cycle = 0; cycle <= 1; cycle++)
		{
			String general_data_partition = PATH + categories[cycle];
			
			File directory = new File( general_data_partition );
			GlobalDict.parse_files( directory , GLOBO_DICT  );
		}
		
		
		for(int cycle = 0; cycle <= 1; cycle++)
		{
			String general_data_partition = PATH + categories[cycle];
					
			File directory = new File( general_data_partition );

			EvaluateFiles.store_file_words_with_label( directory , GLOBO_DICT , train__list_of_file_words );	
		}
		
		
		
		
		EvaluateFiles.generate_word_frequency_count_struc( GLOBO_DICT, train__list_of_file_words, train_freq_count_against_globo_dict );
		
		

		
		
		//Test
		for(int cycle = 0; cycle <= 1; cycle++)
		{			
			//get the test data
			//String test_path = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/data/test";
			String test_path = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/practice_data/data/test";
			File test_dict = new File( test_path + categories[cycle]);

			EvaluateFiles.store_file_words_with_label( test_dict , GLOBO_DICT , test__list_of_file_words );	
		}
		
		EvaluateFiles.generate_word_frequency_count_struc( GLOBO_DICT, test__list_of_file_words, test_freq_count_against_globo_dict );
		
		

		
		GradientDescent.gradient_descent(train_freq_count_against_globo_dict, test_freq_count_against_globo_dict, GLOBO_DICT);

		
		
	}

}
