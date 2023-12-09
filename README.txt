DISCLAIMER: The test case that is described in this README has elements of random number generation. That means that
at step 9, there is ~5% chance that you will trigger a bug that has purposely been placed in the script. If this occurs,
just rerun steps 7-9 until you do not trigger a bug. Additionally, in step 28, there is ~2% chance that you will
not trigger the bug when running these test cases. Again, if this happens, you can just rerun steps 24-28.

This README will guide you through how to use our CGAN to detect a bug in the Buggy_test_Case.py script.

1) This readme assumes the following:

2) You're running this script on Windows.

4) You're running on AMD64 architecture. This may also work on x64, we don't know, we al have Windows computers.
This does not work on arm64.

5) You have venv, Python, and pip installed.

If you don't have Python, venv, or pip installed, follow these instructions for your platform:
https://phoenixnap.com/kb/how-to-install-python-3-windows

1) Open the windows command prompt.

2) CD into the CPRE 513 Final Project folder that this README is contained in.

3) Once this is all setup, find the requirements.txt file in the same folder as this file.

4) Run the command "python -m venv <environment_name>" replacing <environment_name> with any name.

5) Run the command ".\<environment_name>\Scripts\activate" (This is the command on Windows. If you're using another platform, Google is your friend for how to actiavte a venv environment on that platform)

6) Run the command "pip install -r requirements.txt"

7) Run the command "python "Test Case Code/Buggy_test_Case.py""

8)Tell the script you would like to run it using random inputs, by entering "2" then pressing enter.

9) What this command does is it runs a buggy program 100 times, with 100 random inputs between 0 and 1000.
It then saves the path that each input took, as well as the input that caused that path, and then saves
this information to a CSV called Buggy_test_case.csv in the Training Data folder.

10) Now, we will use this data to train the CGAN.

11) Run the command "python "CGAN/CGAN_training.py""

12) When the script prompts you, enter the path for the training data you want to use 
(this will be Training Data/Buggy_test_case.csv)

13) The script will then tell you how many data points this training data has. The data we've just generated has
100 data points. The script will ask you what batch size you want to use. 16 is a good number for this.

14) The script will now ask you how many epochs you want to run. 100 is a good number for this.

15) The script will now ask you what you want to name your generator. Name it Buggy_generator.

16) Once the generator has finished training, you will have a new file called Buggy_generator.pt.

17) Run the command "python "CGAN/CGAN_implementation.py""

18) Give the path to the Buggy_generator.pt file, when it prompts you.(The Buggy_generator.pt file will be in the 
CPRE 513 Final Project folder)

19) Now, give the path to the training data you used to train this model. 
This path will be Training Data\Buggy_test_case.csv.

20) Now, the script will ask you for a path to a file of path requests. This is a file with a list of paths that you're
asking the model to generate input for.

21) Give the model the path: 
Path requests for generator testing\Sample_test_cases_for_model_evaluation_of_buggy_test_case.csv

22) Now, the script will ask you what you want the output file to be called for these path requests. Name it 
Buggy_generator_output.

23) You will now have a file called "Buggy_generator_output.csv". If you look at this file, you will see it is a CSV
with a path column, and an input column. This file is essentially a list of inputs that the generator believes
will take the path "1;1", that is, True, True. We do the path True, True becuase we know that we have a potential
security risk along this path (the random number generation) so we want to throughly test this path to ensure
there is no security risk.

24) Now that you have generated this input, run the command "python "Test Case Code/Buggy_test_case.py"" again.

25) This time, tell it you want to read its inputs from a CSV file, by entering a "1".

26) Give it the path to the CSV file. This will be the path "Buggy_generator_output.csv".

27) The script will now run through 4000 artifically generated inputs, that are likely to take the path "True, True".

28) This means the script will take a path with a security risk 4000 times, hopefully testing it to the point that
any bugs will be revealed. You will most likely get a divide by zero error when running these test cases.


