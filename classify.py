import os
import sys
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader, Instances
from weka.classifiers import Classifier, Evaluation
import time
from weka.filters import Filter
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.clusterers import Clusterer, ClusterEvaluation

class cw3_classifier():
    def __init__(self):
        pass
        # jvm.start()

    def load_data(self, filename, filter=False):
        self.filename = filename
        print("\nLoading dataset: " + filename)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        data.class_is_first()
        if(filter):
            data = self.filter_data(data)
        self.training_data = data

    def load_data_split(self, filename, validation_split, filter=False):
        self.validation_split = validation_split    
        self.filename = filename
        print("\nLoading dataset: " + filename)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        data.class_is_first()
        if(filter):
            data = self.filter_data(data)
        train, test = data.train_test_split(self.validation_split, Random(1))
        self.training_data = train
        self.testing_data = test
    
    def run_crossval(self, output_directory, classifier_name, classifier_weka_spec, options_list):
        # build classifier
        print("\nBuilding "+classifier_name+" Classifier on training data.")
        buildTimeStart=time.time()
        cls = Classifier(classname=classifier_name, options=options_list)
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = classifier_name+" Cross Eval Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nCross Evaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl = Evaluation(self.training_data)
        evl.crossvalidate_model(cls, self.training_data, 10, Random(1))

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = classifier_name+" Cross Eval Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results(classifier_name+"_Crossval",resultsString,output_directory)

    def run_naive_bayes_split(self, output_directory):
        # build classifier
        print("\nBuilding Classifier on training data.")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "NB Split Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nNB Split Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("Naive_Bayes",resultsString,output_directory)

    def save_results(self, classifier, string, output_directory, bif=False):
        try:
            os.mkdir(output_directory)
        except:
            print("Directory Exists, Continuting.\n")
        
        if(bif):
            output_file_path = os.path.join(output_directory,classifier+"_results.bif")
        else:
            output_file_path = os.path.join(output_directory,classifier+"_results.txt")

        try:
            output_file = open(output_file_path,"x")
        except:
            os.remove(output_file_path)
            print("Removed exisiting file\n")
            output_file = open(output_file_path,"x")

        output_file.write(string) 
        output_file.close()
        print("**** Results saved to :"+output_file_path)

    def print_both(self,print_string, resultsString):
        print(print_string)
        resultsString += print_string
        return resultsString

    def filter_data(self,data):
        print("Filtering Data..\n")
        flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
        aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
        assearch = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
        flter.set_property("evaluator", aseval.jobject)
        flter.set_property("search", assearch.jobject)
        flter.inputformat(data)
        filtered = flter.filter(data)
        return filtered

class cw3_helper():
    def __init__(self, start=True):
        if(start):
            #increased to 4gb for bayes network.
            jvm.start(max_heap_size="6g")

    def cleanup(self):
        jvm.stop()