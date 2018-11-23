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

#part1
filename1 = "fer2018/transformed_arffs/fer2017-training-cfs.arff"
filename2 = "fer2018/transformed_arffs/fer2017-testing-cfs.arff"
filename_full = "fer2018/transformed_arffs/fer2017-full-cfs.arff"

# testName = "part1_minimal_binary"
# testName = "part1_minimal_pruning"
# testName = "part1_minimal_confidence"
testName = "j48/training30_testing70/num_instances"

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


def run_classifiers():

    jvm_helper = classify.cw3_helper(False)
    jvm_helper.cleanup()

    threads = []

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
    thread3 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "3"])
    thread4 = myThread(4, "run_j48_holdout", run_j48_holdout, ["-C", "0.25", "-M", "4"])

    # Start new Threads
    # thread1.start()
    # thread2.start()
    thread3.start()
    thread4.start()

    # Add threads to thread list
    # threads.append(thread1)
    # threads.append(thread2)
    threads.append(thread3)
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


    