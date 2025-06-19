This is the user guide of running the antimicrobial peptide prediction model.

First of all, download the Python source code, dataset and pretrained model from https://github.com/researchprotein/amp

Next, download the feature files from https://pan.baidu.com/s/1Z98LXAKs9gmvaQ4-Z3dQsg with password hiur

Subsequently, execute following commands:

unzip amp.zip

mv training\ features.zip amp/

mv independent\ test\ features.zip amp/

cd amp

unzip training\ features.zip

unzip independent\ test\ features.zip

pip install -r requirements.txt

python ensemble.py

And now the program should be running and will output the prediction result for the independent test dataset.
