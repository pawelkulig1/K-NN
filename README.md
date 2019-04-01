* Prequisities:
	* [One needs to have all required libraries installed to run our k-nn project.]
	* [We recommend using python-virtualenv package to install required packages.]
	* [All packages needed to run project are listed in file requirements.txt]
	* [Python version: 3.6 or higher.]

1. To setup virtualenv: (linux/mac)
	1. pip install virtualenv
	2. virtualenv python=python3 venv
	3. . venv/bin/activate

2. To install requirements automatically:
	1. Run virtualenv in project directory
	2. pip install -r requirements.txt

3. Running tests: (inside virtualenv)
	1. python tests.py

4. Running graph:
	1. python graph.py - graphic show woking algorithm
	
5. Files:
	1. knn.py - main algorithm
	2. normalizer.py - data normalization script
	
All data taken from scikitLearn library and our score compared to their algorithms in tests.