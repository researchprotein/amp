This is the user guide of running the antimicrobial peptide prediction model.

First of all, download the Python source code, dataset and pretrained model from https://github.com/researchprotein/amp

Next, execute following commands:

unzip AMP.zip

cd AMP

pip install -r pip requirements.txt

python ensemble.py

And now the program should be running and will output the prediction result for the independent test dataset.
