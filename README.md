# Housing Prices Project #
Summary of results may be found here:
https://docs.google.com/document/d/17UUAweMWBC5K7yGNm9FLFASn2LdhuHec4Ew-loJSExs/edit?usp=sharing

## Installation ##
To install this package you will need the 'conda', or another virtual environment. 
Conda is the one that I used which now has the name `miniconda` and is downloadable here
https://conda.io/miniconda.html
`conda activate hprice`
* From this directory install all relevant packages:
`pip install -r requirements.txt`
`conda instal jupyter`
* Complile code:
`python setup.py develop`

Non-standard packages installed are:
https://github.com/scikit-learn-contrib/sklearn-pandas

## To Run ##
Given a csv file, with housing prices that we wish to predict, we use the full command:
`python predict_prices.py -i data/single_family_home_values.csv`

## To explore ## 
Please feel free to look at the `improved_pipeline.ipynb`, though realize that it is an exploration-style notebook and not intended for presentation purposes.



