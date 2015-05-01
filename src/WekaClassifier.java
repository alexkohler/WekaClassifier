import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SGD;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.MultiClassClassifierUpdateable;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddValues;
/*
 * Note that working jar is the weka.jar in workspace/Face-Tracking/ikvm-7.2.4630/bin
 */


public class WekaClassifier {



	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		ArrayList<Instance> persistedData = new ArrayList<Instance>();
		boolean fromCmd = false;
		String trainDataPath = "data/testFormatNormalized.dat";
		String testDataPath = "data/multiClassData.arff";
		if (args.length > 0) {
//			 System.out.println("Running from cmd");
			 trainDataPath = args[0];
			 testDataPath = args[1];
			 fromCmd = true;
			}
		/** The error correction modes */
		final Tag [] TAGS_METHOD = {
				new Tag(weka.classifiers.meta.MultiClassClassifier.METHOD_1_AGAINST_ALL, "1-against-all"),
				new Tag(weka.classifiers.meta.MultiClassClassifier.METHOD_ERROR_RANDOM, "Random correction code"),
				new Tag(weka.classifiers.meta.MultiClassClassifier.METHOD_ERROR_EXHAUSTIVE, "Exhaustive correction code"),
				new Tag(weka.classifiers.meta.MultiClassClassifier.METHOD_1_AGAINST_1, "1-against-1")
		};
		//*********************************************************
		// THIS IS ALL SET UP STUFF 
		//*********************************************************
		// load data
		// DO THIS ONCE
		ArffLoader loader = new ArffLoader();
		MultiClassClassifier mcc = new MultiClassClassifier();

		mcc.setMethod(new SelectedTag(weka.classifiers.meta.MultiClassClassifier.METHOD_1_AGAINST_ALL, TAGS_METHOD));
		/*Trying various statistics to attempt to get some accurate classification... No luck though, so this may not be the problem */
		LibSVM svm = new LibSVM();
		//svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_ONE_CLASS_SVM, LibSVM.TAGS_SVMTYPE));
//		svm.setDebug(false);
//		svm.setNormalize(true);
//		svm.setProbabilityEstimates(true);
//		mcc.setDebug(false);
//		mcc.setClassifier(svm);
		
		
		IBk ibk = new IBk();
		ibk.setKNN(1);
		ibk.setDistanceWeighting(new SelectedTag(ibk.WEIGHT_INVERSE, ibk.TAGS_WEIGHTING));
		mcc.setClassifier(ibk);
		//mcc.buildClassifier(structure);



		//Train with some actual data...
		loader.setFile(new File(testDataPath));//contains three instances of Alex


		Instance current;

		//mcc.buildClassifier(structure);
//		System.out.println("Training with ");
		Instances insts = loader.getDataSet();
		Instance goodData = insts.get(0);
		insts.setClassIndex(64);
		/*while ((current = loader.getNextInstance(structure)) != null) {
	    		System.out.println(current);
	    		goodData=current;
	    	   persistedData.add(current);//we need to keep track of data we trained with because the classifier gets untrained when we add a new instance (someone who was identified as unknown) 
	    }*/

		mcc.buildClassifier(insts);
		
		//****************************************
		//SETUP ENDS HERE
		//****************************************
		
		//********************************************************************
		//Try to salvage the server fork code from here
		//******************************************************************
		
		

		//Test with some actual data...

		// set class attribute
		Instance badData = null;
		// create copy
//		System.out.println("Testing with ");
			DenseInstance testInstance = new DenseInstance(65);
			File file = new File(trainDataPath);
			Scanner scnr = new Scanner(file);
			String[] vals = scnr.nextLine().split(",");
			testInstance.setDataset(insts);
			for (int j = 0; j < 64; j++)
			{
//				System.out.println(vals[j]);
				testInstance.setValue(j, Double.valueOf(vals[j]));//this test instance is simply a bunch of zeroes (and should be identified as unknown)
			}
			//testInstance = (DenseInstance) unlabeled.get(i);
//			System.out.println(testInstance);

			//gets individual probabilities***************************
			double values[] = mcc.individualPredictions(testInstance);
			//*****************************************
//			for (int j = 0; j < values.length; j++)
//			{
//				System.out.println(values[j]);
//			}
			
			//***************************
			//GET CLASS LABEL
			//********************
			double clsLabel = mcc.classifyInstance(testInstance);//classifier returns an index to the class that it thinks is identified	
			String label = insts.classAttribute().value((int) clsLabel);
//			System.out.println("class label is " + label);
			//************************
			//HERE IS UPDATE CODE
			//**********************
			if (values[(int) clsLabel] < .2) //if we get an unknown class, we need to update the classifier, this is likely where shit is going wrong. 
			{
//				System.out.println(values[(int) clsLabel]);
//				System.out.println("Updating classifier");
				//then add to structure
				
				//*****************8
				//FILTER ADDS A NEW CLASS LABEL to instances, YOU MUST DO THIS
				//**************************
				AddValues f = new AddValues();
				f.setAttributeIndex("last");
				f.setLabels("unknown");//new class label
				f.setInputFormat(insts);
				insts=Filter.useFilter(insts,f);
				//********************************8
				//insts has now been updated
				//******************************
				testInstance.setDataset(insts);
				testInstance.setValue(64, "unknown");//Chadwick is our bullshit class, that is, the instance with nothing but zeroes
				badData = testInstance;//Chadwick is bad data
				//mcc.buildClassifier(insts);//resetting everything?
				insts.add(testInstance);
				//add everything else back in again.
				//take the previous data 
				for (Instance cls : persistedData){
//					System.out.println("adding instance");
					cls.setDataset(insts);
				}
				//REBUILD CLASSIFIER********************8
				mcc.buildClassifier(insts);
				//**************************************

			}
			//******************************
			//UPDATE CODE ENDS HERE
			//******************************
		goodData.setDataset(insts);
		clsLabel = mcc.classifyInstance(goodData);
		label = insts.classAttribute().value((int) clsLabel);
//		System.out.println("good class label is " + label);
//		System.out.println(insts.classAttribute().value((int)goodData.classValue()));
		if (badData != null) {
			badData.setDataset(insts);
			clsLabel = mcc.classifyInstance(badData);
			label = insts.classAttribute().value((int) clsLabel);
			double[] retVal = mcc.distributionForInstance(badData); //if unclassified, returns array of size 0 
			System.out.println(label + "(" + retVal[(int) clsLabel] + ")"); 

/*			for (double d : retVal) 
				System.out.println(d);*/
		}




		//Grab a csv, turn it into an arff, save it
		/*	    System.out.println("Testing class update");
	    	    CSVLoader csvloader = new CSVLoader();
	    	    csvloader.setSource(new File("/home/alex/workspace/c#/wekaFun/wekaFun/facesFormatSource.csv"));
	    	    Instances data = csvloader.getDataSet(); //multiple instances #wat
	    	    ArffSaver saver = new ArffSaver();
	    	    saver.setInstances(data);
	    	    saver.setFile(new File("/home/alex/workspace/c#/wekaFun/wekaFun/actual_data.arff"));
	    	    saver.writeBatch();*/

		//http://comments.gmane.org/gmane.comp.ai.weka/7806
		// 	   https://weka.wikispaces.com/Multi-instance+classification

	}
}