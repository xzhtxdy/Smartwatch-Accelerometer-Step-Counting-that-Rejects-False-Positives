# To-do
- [ ] Peak detection method for step counting;
- [ ] Trainning of the Weighted sampling-based bagging with oneclass support vector machine ensemble;


## Instructions

1 **Extract features**

   Run the script to extract and save features:

     python scripts/extract_and_save_features.py

2 **Train models**

   Generate and train models using the preprocessed features:

     python notebooks/step_counting/generate_models.py

3 **Test step counting**

   Run step counting tests on new walking data:

     python notebooks/step_counting/step_counting_test.py
