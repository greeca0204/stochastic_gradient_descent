package stochastic_gradient_descent;

import java.io.BufferedOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;


public class GradientDescent 
{
	  static int MAX_ITER = 10000000;
	  static double LEARNING_RATE = 0.5;           
	  
    static final String LABEL = "atheism";
    //static final String LABEL = "sports";
    //static final String LABEL = "science";
	  
	  public static void gradient_descent( Table< int[] , String , Integer > train_freq_count_against_globo_dict,
			  					           Table< int[] , String , Integer > test_freq_count_against_globo_dict,
			  						       Set<String> GLOBO_DICT ) throws FileNotFoundException
	  {
		  
		  int globo_dict_size = GLOBO_DICT.size();
		  int number_of_files__train = train_freq_count_against_globo_dict.size();
		  
		  double[] theta = new double[ globo_dict_size + 1 ];//one for bias
		  for (int i = 0; i < globo_dict_size; i++) 
		  {
			  theta[i] = randomNumber(0,1);
		  }
		  theta[ globo_dict_size ] = randomNumber(0,1);
		    
		  		    

		   double[][] feature_matrix__train = new double[ number_of_files__train ][ globo_dict_size ];
		   double[] outputs__train = new double [ number_of_files__train ];
		    
		   int z = 0;
		   for ( Cell< int[] , String , Integer > cell: train_freq_count_against_globo_dict.cellSet() )
		   {			   
			   int[] container_of_feature_vector = cell.getRowKey();
			   
			   for (int q = 0; q < globo_dict_size; q++) 
	           {
				   feature_matrix__train[z][q] = container_of_feature_vector[q];

	           }
			   // use 1 and -1 not 0 and 1
			   outputs__train[z] = String.valueOf( cell.getColumnKey() ).equals(LABEL) ? 1.0 : -1.0;
	           
	           z++;
		   }
	
	  double hypothesis;
	  int p;
	  
	  boolean misclassified;
	  
	  /* gradient: one for each file? one for each weight? or one for each epoch? */
	  double[] gradient = new double[ globo_dict_size + 1 ];
	  
	  
	  do 
	  {
	    
		misclassified = false;
		    
		Arrays.fill(gradient, 0);
		  
		//shuffle
	     
	    // iterate in a random order
	    // random_index_array[number_of_files_train]
	    ArrayList <Integer> random_index_array = new ArrayList<Integer>();
	    
	    for( int i = 0; i < number_of_files__train; i++)
	    {
	    	random_index_array.add(i); 
	    }	
	    // shuffle
	    Collections.shuffle( random_index_array );
	    	
		for (p = 0; p < number_of_files__train; p++) 
		{
			  //access randomly
			  int idx = random_index_array.get(p);	  
	    
			  hypothesis = calculateHypothesis( theta, feature_matrix__train, idx, globo_dict_size );
			  
			  // missclassified or not
			  if ( hypothesis * outputs__train[idx] <= 1)
			  {
				  misclassified = true;
				  
					 for (int i = 0; i < globo_dict_size; i++) 
					 {
						 gradient[i] += ( outputs__train[idx] * feature_matrix__train[idx][i] );
					 }
					 //bias
					 gradient[globo_dict_size] += outputs__train[ idx ];
			  }

		  }
	      
		  
    	  for (int i = 0; i <= globo_dict_size; i ++)
    	  {
    		  theta[i] += LEARNING_RATE * gradient[i];
    	  }


	  }
	  while(misclassified);
	

	  
	  
	  
      int number_of_files__test = test_freq_count_against_globo_dict.size();
      double[][] feature_matrix__test = new double[ number_of_files__test ][ globo_dict_size ];
       
	   //i don't actually need this info, but something to clarify the output would be great
	   String[] test_file_true_label = new String [ number_of_files__test ];
	    
	   int x = 0;
	   for ( Cell< int[] , String , Integer > cell: test_freq_count_against_globo_dict.cellSet() )
	   {			   
		   int[] container_of_feature_vector__test = cell.getRowKey();
		   //System.out.println( Arrays.toString( container_of_feature_vector ) );
		   
		   for (int q = 0; q < globo_dict_size; q++) 
          {
			   feature_matrix__test[x][q] = container_of_feature_vector__test[q];
          }
		   test_file_true_label[x] = (String)( cell.getColumnKey() );
          
          x++;
	   }
	   //System.out.println( Arrays.toString( outputs ) );
	   System.out.println();
	   
	   
	   double tp = 0.0;
	   double fp = 0.0; 
	   double tn = 0.0;
	   double fn = 0.0;
	   
	  for (p = 0; p < number_of_files__test; p++) 
	  {
		  double predicted_class = calculateHypothesis( theta, feature_matrix__test, p, globo_dict_size );
	      System.out.println("predicted class = " + predicted_class );
	      
	      int actual_class = ( test_file_true_label[p] ).equals(LABEL) ? 1 : -1;
	      
	      System.out.println( "actual class = " + actual_class );
	      
	      System.out.println( "actual class = " + test_file_true_label[p] );
	      
	      //System.out.println();
	        
	      if( actual_class > 0 && predicted_class > 0 )
	    	  tp++;
	      if( actual_class > 0 && predicted_class < 0 )
	    	  fn++;
	      if( actual_class < 0 && predicted_class > 0 )
	    	  fp++;
	      if( actual_class < 0 && predicted_class < 0 )
	    	  tn++;  
	  }
	  
	  System.out.println( "tp: " + tp );
	  System.out.println( "fp: " + fp );
	  System.out.println( "tn: " + tn );
	  System.out.println( "fn: " + fn );
	  System.out.println();
	  
	  double precision = tp / (tp + fp);
	  System.out.println( "precision = " + precision );
	  
	  double recall = tp / (tp + fn);
	  System.out.println( "recall = " + recall );
	  
	  double f_measure = ( 2 * ( precision * recall ) ) / ( precision + recall );
	  System.out.println( "f_measure = " + f_measure );
	  
	  System.out.println();
	  System.out.println();

	  
	  
	  
	  
	  
	  }
	  
	  
	  
	  
	static double calculateHypothesis( double[] theta, double[][] feature_matrix, int file_index, int globo_dict_size )
	{
		double hypothesis = 0.0;

		 for (int i = 0; i < globo_dict_size; i++) 
		 {
			 hypothesis += ( theta[i] * feature_matrix[file_index][i] );
		 }
		 //bias
		 hypothesis += theta[ globo_dict_size ];

		 return hypothesis;
	}
	 
	
	public static double randomNumber(int min , int max) 
	{
		DecimalFormat df = new DecimalFormat("#.####");
		double d = min + Math.random() * (max - min);
		String s = df.format(d);
		double x = Double.parseDouble(s);
		return x;
	 } 
	 
	 

}
