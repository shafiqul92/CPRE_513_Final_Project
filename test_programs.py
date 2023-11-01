import pandas as pd
import numpy as np
import csv
import random

# Create your dictionary class
class keyValuePairs(dict):

    # __init__ function
    def __init__(self):
        self = dict()

    # Function to add key:value
    def add(self, key, value):
        self[key] = str(value)


# Simple input: Singular input and branch
def simpleProgram():
  test_list = []
  i = 0
  num_samples = 10000
  while i < num_samples:
    dict_obj = {}
    val = random.randint(0,1000)
    if (val > 500):
      dict_obj[val] = 1
      i += 1
    else:
      dict_obj[val] = 0
      i += 1
    test_list.append(dict_obj)

  with open('test_case0.csv','w') as f:
    for i in test_list:
        f.write(str(list(i.keys())[0]) + "," + str(list(i.values())[0]))
        f.write("\n")


  #for key in dict_obj:
    #print(key, ":", dict_obj[key])
    #print(dict_obj.keys())



# Simple Test: Singular input & two branches
def simpleProgram1():
  test_list = [] 
  i = 0
  num_samples = 10000
  while i < num_samples:
    dict_obj = {}
    val = random.randint(0,1000)
    if (val < 500):
      dict_obj[val] = "1;"
      if (val < 250):
        dict_obj[val] += "1"
        i+=1
      else:
        dict_obj[val] += "0"
        i+=1

    else:
      dict_obj[val] = "0"
      i += 1
    test_list.append(dict_obj)

  with open('test_case1.csv','w') as f:
    for i in test_list:
        f.write(str(list(i.keys())[0]) + "," + str(list(i.values())[0]))
        f.write("\n")

  #for key in dict_obj:
    #print(key, ":", dict_obj[key])

# Simple Test: Singular input & nested branches
def simpleProgram2():
  dict_obj = keyValuePairs()
  i = 0
  num_samples = 1000
  while i < num_samples:
    dict_obj.key = i
    if (i < 750):
      dict_obj.value = "1"
      if (i == 0):
        dict_obj.value += "1"
        dict_obj.value += "/"
        dict_obj.add(dict_obj.key, dict_obj.value)
      if (i > 0 & i < 600):
        dict_obj.value += "1"
        if (i < 400):
          dict_obj.value += "1"
          if (i > 50 and i < 299):
            dict_obj.value += "1"
            dict_obj.value += "/"
          else:
            dict_obj.value += "0"
            dict_obj.value += "/"
        else:
            dict_obj.value += "0"
            dict_obj.value += "/"
        dict_obj.add(dict_obj.key, dict_obj.value)

      else:
        dict_obj.value += "0"
        dict_obj.value += "/"
        dict_obj.add(dict_obj.key, dict_obj.value)

      i+=1

    else:
      dict_obj.value = "0"
      dict_obj.value += "/"
      dict_obj.add(dict_obj.key, dict_obj.value)
      i += 1

  #for key in dict_obj:
    #print(key, ":", dict_obj[key])

# Simple Programs: while loops
def simpleProgram3():
  dict_obj = keyValuePairs()
  i = 0
  num_samples = 10
  while i < num_samples:
    dict_obj.key = i #sample index
    k = 0
    while k < 50:  #change this while loop condition for more complex loop behaviors
      if (k == 0) : #treat this branch as an initializer for the dictionary value
          dict_obj.value = "1"
          k += 1

      dict_obj.value += "1" #keep count of branch taken/re-entering the while loop
      k += 1

    dict_obj.value += "0" #exit component
    dict_obj.value += "/"
    dict_obj.add(dict_obj.key, dict_obj.value)
    i += 1  #increment to next sample

  #for key in dict_obj:
    #print(key, ":", dict_obj[key])

# Simple Programs: Nested Branches

def simpleProgram4():
  print("test")
  dict_obj = keyValuePairs()
  i = 0
  num_samples = 10
  while i < num_samples:
    dict_obj.key = i
    value = random.randint(0,100)
    if value > 50:
      dict_obj.value = "1"
      if value > 75:
        dict_obj.value += "1"
        if value > 90:
            dict_obj.value += "1"
        else:
            dict_obj.value += "0"
      else:
        dict_obj.value += "0"
        if value > 60:
          dict_obj.value += "1"
        else:
            dict_obj.value += "0"
    else:
      dict_obj.value = "0"
      if value > 25:
        dict_obj.value += "1"
        if value > 40:
          dict_obj.value += "1"
        else:
            dict_obj.value += "0"
      else:
        dict_obj.value += "0"
        if value > 10:
          dict_obj.value += "1"
        else:
          dict_obj.value += "0"  
    dict_obj.value += "/"
    dict_obj.add(dict_obj.key, dict_obj.value)
    i += 1  #increment to next sample

  #for key in dict_obj:
    #print(key, ":", dict_obj[key])

def main():
    simpleProgram()
    simpleProgram1()

if __name__=='__main__':
    main()