import java.io.File;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class TrafficSignDectector {

	
	public static void main(String[] args) {
		System.out.println("Processing started...");
		TrafficSignDectector imageClassifierapp = new TrafficSignDectector();
		//file paths & source file names
		String filepath="/Users/sethulekshmy/eclipse-workspace/DataMiningProject/data/";
		String xfileName = "x_train_gr_smpl.csv";
		String yfileName = "y_train_smpl.csv";
		String outputfile = "merged_xy_train_gr_smpl.arff";
		
		
		 Instances dataSet, dataSet_cls_lbl = null;
		 try {
			// Step 1 - Read data into weka Instances from the CSV file.
			dataSet = imageClassifierapp.getInstancesFromCsvFile(filepath+xfileName);
			System.out.println("Attribute count of dataSet= "+dataSet.numAttributes());
			
			//Step 2- Read the class label file (from Y file) and merge with above Instance using filter	
			dataSet_cls_lbl = imageClassifierapp.getInstancesFromCsvFile(filepath+yfileName);
			
			// Step 3- merging attributes from Y file onto existing Instances created from X file.. 
			//The class label is added as last attribute to the current Instance.
		    Add filterAdd = new Add();
		    filterAdd.setAttributeIndex("last");
		    filterAdd.setAttributeName("sign");
		    filterAdd.setInputFormat(dataSet);
		    dataSet = Filter.useFilter(dataSet, filterAdd);
	        
	        for (int i = 0; i < dataSet.numInstances(); i++) {
	        	dataSet.instance(i).setValue(dataSet.numAttributes() - 1, dataSet_cls_lbl.instance(i).value(0));
	        }
	        
	        System.out.println("Attribute count of dataSet after merging class label from Y file= "+dataSet.numAttributes());
	        
			//Step 4 - convert the Instances into a ARFF format and write to output file
			imageClassifierapp.convertInstancesToArffFile(dataSet, filepath+outputfile);
			System.out.println("Merged ARFF file is saved to: "+filepath+outputfile);
			
			// Step 5 - Apply Filter Numeric to Nominal on last attribute (class)
			dataSet = imageClassifierapp.FilterNumbericToNominal(dataSet,"last");
			
			// Step 6 - Setting last attribute as the class
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
			
			// Step 7 - Apply Filter Best Attribute selection
			dataSet = imageClassifierapp.FilterBestAttributeSelector(dataSet);
			
			// Step 8 - Write filtered data to arff file
	        String outputfilefiltered = "filtered_xy_train_gr_smpl.arff";
	        imageClassifierapp.convertInstancesToArffFile(dataSet, filepath+outputfilefiltered);
			System.out.println("Filtered ARFF file is saved to: "+filepath+outputfilefiltered);
			
			//Step 9 - Apply Classifiers
			
			// BayesNet using TAN
			imageClassifierapp.ApplyClassifiers(dataSet, "BayesNet", "TAN");
			
			// BayesNet using K2
			imageClassifierapp.ApplyClassifiers(dataSet, "BayesNet", "K2");
			
			// NaiveBayes
			imageClassifierapp.ApplyClassifiers(dataSet, "NaiveBayes", "");
			
	        
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 
		 

	}
	
	/**
	*  This method will read data from CSV file and returns the weka Instances
	* 
	* @param file
	*/
	public Instances getInstancesFromCsvFile(String infile) throws Exception {
		
		// build Instances from CSV files
	    CSVLoader loader = new CSVLoader(); 
	    loader.setSource(new File(infile));
	    
	    Instances instances = loader.getDataSet();
	    return instances;    

	}
	 
	 /**
	 *  This method will write Instances data to ARFF file
	 * 
	 * @param file
	 */
	 public void convertInstancesToArffFile(Instances instances, String outfile) throws Exception {
		
		// convert Instances to ARFF file
		ArffSaver arffsaver = new ArffSaver();
		arffsaver.setInstances(instances);
		arffsaver.setFile(new File(outfile));
		arffsaver.writeBatch();    

	 }
	 
	 /**
	 *  This method will apply numeric to nominal filter
	 * 
	 * @param instances
	 * @param attributeIndex
	 */
	 public Instances FilterNumbericToNominal(Instances instances, String attributeIndex) throws Exception {
		
		//Applying Filter - Numeric to Nominal
        System.out.println("Applying Filter - Numeric to Nominal filter on attribute index "+attributeIndex);
        
        Instances nominalData = null;
        nominalData = new Instances(instances);
        
        NumericToNominal filterNominal= new NumericToNominal();
        filterNominal.setOptions(new String[] { "-R", attributeIndex});
        filterNominal.setInputFormat(nominalData);
        nominalData=Filter.useFilter(nominalData, filterNominal);  
        return nominalData;

	 }
	 
	 /**
	 *  This method will apply filter - Best Attribute Selector
	 * 
	 * @param instances
	 */
	 public Instances FilterBestAttributeSelector(Instances instances) throws Exception {
		
		//Applying Filter - Best Attribute selection
        System.out.println("Applying Filter - Best Attribute selection");
        
        Instances nominalData = null;
        nominalData = new Instances(instances);
        
        AttributeSelection filterAttributeSelection = new AttributeSelection();
	    
	    CfsSubsetEval eval = new CfsSubsetEval();
	    eval.setOptions(new String[] { "-P", "1" , "-E", "1"});
        
	    BestFirst search = new BestFirst();
	    search.setOptions(new String[] { "-D", "1" , "-N", "5"});
	    
	    
	    filterAttributeSelection.setEvaluator(eval);
	    filterAttributeSelection.setSearch(search);
	    
	    filterAttributeSelection.setInputFormat(nominalData);
	    nominalData=Filter.useFilter(nominalData, filterAttributeSelection);
	    
	    System.out.println("Best attribute count="+nominalData.numAttributes());
	    
	    return nominalData;

	 }
	 
	 /**
	 *  This method will apply filter - Best Attribute Selector
	 * 
	 * @param instances
	 */
	 public void ApplyClassifiers(Instances instances, String strClassifierName, String classifierOption) throws Exception {
        
		System.out.println("Training Model using classifier: "+strClassifierName+ " with options: "+classifierOption);
		
		Evaluation evaluation = null;
        if ("BayesNet".contentEquals(strClassifierName)) {
        	
        	BayesNet classifier = new BayesNet();
        	if ("TAN".contentEquals(classifierOption)) {
        		System.out.println("Applying Classifier - BayesNet with TAN search and SimpleEstimator");
        		classifier.setOptions(new String[] { "-D", "-Q", "weka.classifiers.bayes.net.search.local.TAN", "--", "-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A","0.5"});
        	} else if ("K2".contentEquals(classifierOption)) {
        		System.out.println("Applying Classifier - BayesNet with K2 search and SimpleEstimator");
        		classifier.setOptions(new String[] { "-D", "-Q", "weka.classifiers.bayes.net.search.local.K2", "--", "-P", "1", "-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A","0.5"});
        	    
        	}
    	    
    	    classifier.buildClassifier(instances);
    	    
    	    // Cross Validation = 10fold,  Seed=1
    	    evaluation = new Evaluation(instances);
    	    evaluation.crossValidateModel(classifier, instances, 10, instances.getRandomNumberGenerator(1));
        
        } else if ("NaiveBayes".contentEquals(strClassifierName)){
        	System.out.println("Applying Classifier - NaiveBayes");
        	NaiveBayes classifierNB = new NaiveBayes();
        	classifierNB.buildClassifier(instances);
    	    
    	    // Cross Validation = 10fold,  Seed=1
    	    evaluation = new Evaluation(instances);
    	    evaluation.crossValidateModel(classifierNB, instances, 10, instances.getRandomNumberGenerator(1));
    	    
        } 
	      
	    
	    StringBuffer result;

	    result = new StringBuffer();
	    result.append("******* Results *********\n");
	     try {
	    	 result.append(evaluation.toSummaryString() + "\n");
	    	 result.append(evaluation.toMatrixString() + "\n");
	         result.append(evaluation.toClassDetailsString() + "\n");
	        
	      } catch (Exception e) {
	        e.printStackTrace();
	      }

	      System.out.println("Classifier Outputs\n"+result.toString());
	    
	    
	  }

}
