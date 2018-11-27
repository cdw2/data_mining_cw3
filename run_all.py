#!./weka_env/bin/python3
import os
import sys
import csv_arff
import part3_attributeReduction
import extract_pixels
import classify
import threading

csv_folder="fer2018/csv"
arff_folder="fer2018/arffs"
reduction_folder="fer2018/reduced_arffs"
transformed_arffs="fer018/transformed_arffs"
pixel_values="pixel_values"

preprocess = False
if(len(sys.argv)==2):
    if sys.argv[1]=="--preprocess":
        preprocess=True

#part2
filename1 = "fer2018/transformed_arffs/fer2017-training-neural-cfs.arff"
filename2 = "fer2018/transformed_arffs/fer2017-testing-neural-cfs.arff"
filename_full = "fer2018/transformed_arffs/fer2017-neural-cfs.arff"

#Part 1
# testName = "part1_minimal_binary"
# testName = "part1_minimal_pruning"
# testName = "part1_minimal_confidence"

# experiemnt_name = "cross_validation"
# experiemnt_name = "train_test"
# experiment_name = "train70_test30"
experiment_name = "train30_test70"

#Part 2
# testName = "neural_network/"+experiment_name+"/learning_rate"
# testName = "neural_network/"+experiment_name+"/momentum"
# testName = "neural_network/"+experiment_name+"/num_epochs"
# testName = "neural_network/"+experiment_name+"/num_layers"
# testName = "neural_network/"+experiment_name+"/num_neurons"
testName = "neural_network/"+experiment_name+"/validation"

class myThread (threading.Thread):
   def __init__(self, threadID, name, function, args=None):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.function = function
      self.args = args
   def run(self):
      print ("Starting " + self.name)
      self.function(self.args)

def convert_to_arff():

    print("*** Converting CSVs ****\n")
    filenames = os.listdir(csv_folder)

    for csv in filenames:
        filename=os.path.join(csv_folder,csv)
        print("Converting "+filename)
        
        # print(csv)
        if(csv=="fer2017-training.csv" or csv=="fer2017-testing.csv"):
            csv_converter = csv_arff.Convert(filename,False)
            csv_converter.run()
        else:
            csv_converter = csv_arff.Convert(filename,True)
            csv_converter.run()

def reduce_attr():
    print("\n*** Reducing Arffs ****\n")
    filenames = os.listdir(arff_folder)

    for arff in filenames:
        filename=os.path.join(arff_folder,arff)
        print("Reducing "+filename)
        arff_reducer = part3_attributeReduction.reduce_attributes(filename)
        arff_reducer.run()

def extract():
    print("\n*** Extracting Pixels ****\n")
    filenames = os.listdir(pixel_values)

    for values in filenames:
        filename=os.path.join(pixel_values,values)
        print("Extracting from "+filename)
        extractor = extract_pixels.extract_pix(filename)
        extractor.run()

def run_j48_cross(options):
    global filename1, filename2, testName
    jvm_helper = classify.cw3_helper()

    j48_cross = classify.cw3_classifier()
    j48_cross.load_data_seperate(filename1,filename2)
    j48_cross.run_crossval("results/"+str(testName),"J48","weka.classifiers.trees.J48", options)

def run_j48_holdout(options):
    global filename_full, filename1, filename2, testName
    jvm_helper = classify.cw3_helper()

    j48_cross = classify.cw3_classifier()
    # j48_cross.load_data_seperate(filename1,filename2)
    j48_cross.load_data_split(filename_full, 30)
    j48_cross.run_split("results/"+str(testName),"J48","weka.classifiers.trees.J48", options)

def run_mlp_cross(options):
    global filename_full, filename1, filename2, testName
    jvm_helper = classify.cw3_helper()

    mlp = classify.cw3_classifier()
    # j48_cross.load_data_seperate(filename1,filename2)
    mlp.load_data(filename_full)
    mlp.run_crossval("results/"+str(testName),"MLP","weka.classifiers.functions.MultilayerPerceptron", options)

def run_mlp_split(options):
    global filename_full, filename1, filename2, testName
    jvm_helper = classify.cw3_helper()

    mlp = classify.cw3_classifier()
    mlp.load_data_split(filename_full, 30)
    # mlp.load_data_seperate(filename1,filename2)
    mlp.run_split("results/"+str(testName),"MLP","weka.classifiers.functions.MultilayerPerceptron", options)


def run_classifiers():

    jvm_helper = classify.cw3_helper(False)
    jvm_helper.cleanup()

    threads = []
    thread1 = None
    thread2 = None
    thread3 = None
    thread4 = None

    #Part 1 - Binary Splits
    # thread1 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "2"])
    # thread2 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-B", "-M", "2"])

    #Part 1 - Pruning
    # thread1 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "2"])
    # thread2 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-U", "-M", "2"])

    #Part 1 - Confidence Threshold
    # thread1 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.10", "-M", "2"])
    # thread2 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "2"])
    # thread3 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.35", "-M", "2"])
    # thread4 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.50", "-M", "2"])

    # #Part 1 - Minimum_Number_Of_Instances
    # thread1 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "1"])
    # thread2 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "2"])
    # thread3 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "3"])
    # thread4 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "4"])


    #Part2 - MLP ==========================================================
    mlpfunction = run_mlp_split
    #learning rate
    # thread1 = myThread(1, "run_mlp", mlpfunction, ["-L", "3.0", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread2 = myThread(2, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread3 = myThread(3, "run_mlp", mlpfunction, ["-L", "0.03", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread4 = myThread(4, "run_mlp", mlpfunction, ["-L", "0.003", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])

    # #momentum
    # thread1 = myThread(1, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.1", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread2 = myThread(2, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread3 = myThread(3, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.3", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread4 = myThread(4, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.4", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])

    # #num_epochs
    # thread1 = myThread(1, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread2 = myThread(2, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "200", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread3 = myThread(3, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "300", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread4 = myThread(4, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "400", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])

    # #num_layers
    # thread1 = myThread(1, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread2 = myThread(2, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4,4", "-R"])
    # thread3 = myThread(3, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4,4,4", "-R"])
    # thread4 = myThread(4, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4,4,4,4", "-R"])

    # #num_neurons
    # thread1 = myThread(1, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "1", "-R"])
    # thread2 = myThread(2, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread3 = myThread(3, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "8", "-R"])
    # thread4 = myThread(4, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "12", "-R"])

    # #validation
    # thread1 = myThread(1, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "10", "-H", "4", "-R"])
    # thread2 = myThread(2, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "20", "-H", "4", "-R"])
    # thread3 = myThread(3, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "30", "-H", "4", "-R"])
    # thread4 = myThread(4, "run_mlp", mlpfunction, ["-L", "0.3", "-M", "0.2", "-N", "100", "-V", "0", "-S", "0", "-E", "40", "-H", "4", "-R"])
    
    if thread1 is not None:
        thread1.start()
        threads.append(thread1)
   
    if thread2 is not None:
        thread2.start()
        threads.append(thread2)
        
    if thread3 is not None:
        thread3.start()
        threads.append(thread3)
        
    if thread4 is not None:
        thread4.start()
        threads.append(thread4)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    jvm_helper.cleanup()
    print ("Exiting Main Thread")

try:
    if(preprocess):
        print("***** Preprocessing Data ******")
        convert_to_arff()
        reduce_attr()
        # extract()
    else:
        run_classifiers()

except Exception as e:
    print(e)


    