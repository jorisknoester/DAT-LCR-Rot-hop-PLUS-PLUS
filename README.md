# DAT-LCR-Rot-hop++

Cross-Domain (CD) Aspect-Based Sentiment Classification (ABSC) using LCR-Rot-hop++ with Domain Adversarial Training (DAT).

## Set-up instructions.

- Set-up a virtual environment:
    - Make sure you have a recent release of Python installed (we used Python 3.7), if not download
      from: https://www.python.org/downloads/
    - Download Anaconda: https://www.anaconda.com/products/individual
    - Set-up a virtual environment in Anaconda using Python 3.5 in order to be able to download tensorflow 1.5 package. Newer versions of python are not compatible.
    - Copy all software from this repository into a file in the virtual environment.
    - Open your new environment in the command window ('Open Terminal' in Anaconda)
    - Navigate to the file containing all repository code (file_path) by running: ```cd file_path```
    - Install the requirements by running the following command:
      ```pip install -r requirements.txt```
    - You can open and edit the code in any editor, we used the PyCharm IDE: https://www.jetbrains.com/pycharm/
## How to use?

- Make sure the Python interpreter is set to your Python 3.5 virtual environment (we used PyCharm IDE).
Get raw data for your required domains by running raw$\_$data.py for restaurant, laptop,and  book domain.
-  Adjust the paths in config.py, main$\_$test.py, and main$\_$hyper.py 
- Get BERT embeddings by running files in getBERT for your required domains *using Google Colab* to obtain BERT embeddings (see files for
      further instructions on how to run).
- Prepare BERT train and test file and BERT embedding:
    - Run prepare_bert.py for your required domains.
- Tune hyperparameters to your specific task using main_hyper.py or use hyperparameters as pre-set in main_test.py.
- Select tests to run and run main_test.py (running all tests will take a long time, 4-5 minutes per iteration). Make
  sure write_result is set to True if you want the results to be saved to a text file.

## References.

This code is adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).

https://github.com/mtrusca/HAABSA_PLUS_PLUS

Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128, pp. 365-380. Springer, Cham.
https://doi.org/10.1007/978-3-030-50578-3_25
