import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MultiClassClassifier;
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


public class main {



	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		ArrayList<Instance> persistedData = new ArrayList<Instance>();
//		System.out.println("args are");
		boolean fromCmd = false;
		String dataPath = "/home/alex/workspace/c#/wekaFun/wekaFun/data.arff";
		String trainDataPath = "/home/alex/workspace/c#/wekaFun/wekaFun/new_data.arff";
		String testDataPath = "/home/alex/workspace/c#/wekaFun/wekaFun/testFormat.dat";
		if (args.length > 0) {
//		 System.out.println("Running from cmd");
		 trainDataPath = args[0];
		 testDataPath = args[1];
		 fromCmd = true;
		}
		/* The error correction modes */
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
		loader.setFile(new File(dataPath));
		MultiClassClassifier mcc = new MultiClassClassifier();

		mcc.setMethod(new SelectedTag(weka.classifiers.meta.MultiClassClassifier.METHOD_1_AGAINST_ALL, TAGS_METHOD));
		/*Trying various statistics to attempt to get some accurate classification... No luck though, so this may not be the problem */
		LibSVM svm = new LibSVM();
		//svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_ONE_CLASS_SVM, LibSVM.TAGS_SVMTYPE));
		svm.setDebug(false);
		svm.setNormalize(true);
		svm.setProbabilityEstimates(true);
		mcc.setDebug(false);
		mcc.setClassifier(svm);
		
		
		IBk ibk = new IBk();
		ibk.setKNN(1);
		ibk.setDistanceWeighting(new SelectedTag(ibk.WEIGHT_INVERSE, ibk.TAGS_WEIGHTING));
		mcc.setClassifier(ibk);
		//mcc.buildClassifier(structure);



		//Train with some actual data...
		loader.setFile(new File(trainDataPath));//contains three instances of Alex


		Instance current;

//		System.out.println("Training with ");
		Instances insts = loader.getDataSet();
		Instance goodData = insts.get(0);
		insts.setClassIndex(64);

		mcc.buildClassifier(insts);
		
		//****************************************
		//SETUP ENDS HERE
		//****************************************
		

		//Test with some actual data...
		// set class attribute
		Instance badData = null;
		// create copy

			//Populate data for our test instance
			DenseInstance currentTestInstance = new DenseInstance(65);
			File file = new File(testDataPath);
			Scanner s = new Scanner(file);
			String[] input = s.nextLine().split(",");
			currentTestInstance.setDataset(insts);
			for (int j = 0; j < 64; j++)
			{
				currentTestInstance.setValue(j, /*Double.valueOf(input[j])*/0);//this test instance is simply a bunch of zeroes (and should be identified as unknown)
			}
			s.close();

			//gets individual probabilities***************************
			double values[] = mcc.individualPredictions(currentTestInstance);
			//*****************************************
/*			for (int j = 0; j < values.length; j++)
			{
				System.out.println(values[j]);
			}*/
			
			//***************************
			//GET CLASS LABEL
			//********************
			double clsLabel = mcc.classifyInstance(currentTestInstance);//classifier returns an index to the class that it thinks is identified	
			String label = insts.classAttribute().value((int) clsLabel);
//			System.out.println("class label is " + label);
			//************************
			//HERE IS UPDATE CODE
			//**********************
//			mcc =  (MultiClassClassifier) weka.core.SerializationHelper.read("/home/alex/workspace/c#/wekaFun/wekaFun/classifier.model");

			if (values[(int) clsLabel] < .2) //if we get an unknown class, we need to update the classifier, this is likely where shit is going wrong. 
			{
				//then add to structure
				
				//*****************8
				//FILTER ADDS A NEW CLASS LABEL to instances, YOU MUST DO THIS
				//**************************
				AddValues f = new AddValues();
				f.setAttributeIndex("last");
				s = new Scanner(System.in);
				String name = "unknown";
				s.close();
				f.setLabels(name);//new class label
				f.setInputFormat(insts);
				insts=Filter.useFilter(insts,f);
				//********************************8
				//insts has now been updated
				//******************************
				currentTestInstance.setDataset(insts);
				currentTestInstance.setValue(64, name);//Chadwick is our bullshit class, that is, the instance with nothing but zeroes
				badData = currentTestInstance;//Chadwick is bad data
				//mcc.buildClassifier(insts);//resetting everything?
				insts.add(currentTestInstance);
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
			else { //exists in our classifier!
				clsLabel = mcc.classifyInstance(currentTestInstance);
				label = insts.classAttribute().value((int) clsLabel);
//				System.out.println("Already been identified... This is " + label);
				
				
			}
			//******************************
			//UPDATE CODE ENDS HERE
			//******************************
		goodData.setDataset(insts);
//		clsLabel = mcc.classifyInstance(goodData);
		label = insts.classAttribute().value((int) clsLabel);
		System.out.println(label);
//		System.out.println(insts.classAttribute().value((int)goodData.classValue()));
		if (badData != null) {
			badData.setDataset(insts);
			clsLabel = mcc.classifyInstance(badData);
			label = insts.classAttribute().value((int) clsLabel);
		}
		
		
//		System.out.println("Classifier updated");
		weka.core.SerializationHelper.write("/home/alex/workspace/c#/wekaFun/wekaFun/classifier.model", mcc);



		//Grab a csv, turn it into an arff, save it
/*			    System.out.println("Testing class update");
	    	    CSVLoader csvloader = new CSVLoader();
	    	    csvloader.setSource(new File("/home/alex/workspace/c#/wekaFun/wekaFun/facesFormatSource.csv"));
	    	    Instances data = csvloader.getDataSet(); //multiple instances #wat
	    	    
	    	    ArffSaver saver = new ArffSaver();
	    	    saver.setInstances(data);
	    	    saver.setFile(new File("/home/alex/workspace/c#/wekaFun/wekaFun/sum_data.arff"));
	    	    saver.writeBatch();*/

		//http://comments.gmane.org/gmane.comp.ai.weka/7806
		// 	   https://weka.wikispaces.com/Multi-instance+classification

	}
}