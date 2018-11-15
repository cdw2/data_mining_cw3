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

#task 1.1
filename = "fer2018/reduced_arffs/reduced.arff"
testName = "task1_1"

# #task3
# filename="fer2018/reduced_arffs/fer2018.reduced.arff"
# testName = "task_5_clustering"

# #task5
# filename="fer2018/reduced_arffs/fer2018surprise.reduced.arff"
# testName = "surprise"

#task7
# filename="fer2018/transformed_arffs/transformed_14.arff"
# testName = "task7_14Pixels_seed_test"


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
        
        if(csv=="fer2018.csv"):
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

def run_j48_cross(args):
    global filename, testName
    jvm_helper = classify.cw3_helper()

    j48_cross = classify.cw3_classifier()
    j48_cross.load_data(filename)
    j48_cross.run_crossval("results/"+str(testName),"J48","weka.classifiers.trees.J48", ["-K", "3", "-W"])


def run_classifiers():

    jvm_helper = classify.cw3_helper(False)
    jvm_helper.cleanup()

    threads = []

    #TASK 1
    # Create new threads
    # thread1 = myThread(1, "IBK-Cross-Val", run_ibk_crossval)
    # thread2 = myThread(2, "IBK-Split", run_ibk_split)
    # thread3 = myThread(3, "run_nb_split", run_nb_split)
    # thread4 = myThread(4, "run_nb_crossval", run_nb_crossval)

    #TASK3
    # thread1 = myThread(1, "run_bayes_split", run_bayes_split, (1))
    # thread2 = myThread(2, "run_bayes_split", run_bayes_split, (3))
    # thread3 = myThread(3, "run_bayes_split", run_bayes_split, (5))

    #TASK5
    # thread1 = myThread(1, "run_simplekm_noclass", run_simplekm_noclass)
    # thread2 = myThread(2, "run_simplekm_with_class", run_simplekm_with_class)

    #TASK7
    # thread1 = myThread(1, "run_clusters_auto1", run_clusters_auto1)
    # thread1 = myThread(2, "run_clusters_auto2", run_clusters_auto2)

    # thread1 = myThread(1, "run_clusters_manual", run_clusters_manual, 4)
    # thread2 = myThread(2, "run_clusters_manual", run_clusters_manual, (15))
    # thread3 = myThread(3, "run_clusters_manual", run_clusters_manual, (7))
    # thread4 = myThread(4, "run_clusters_manual", run_clusters_manual, (10))

    thread1 = myThread(4, "run_seeds_manual", run_j48_cross, (0))

    # Start new Threads
    thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()

    # Add threads to thread list
    threads.append(thread1)
    # threads.append(thread2)
    # threads.append(thread3)
    # threads.append(thread4)

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
        extract()
    else:
        run_classifiers()

except Exception as e:
    print(e)


    