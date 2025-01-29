# E2E-FS
E2E-FS: An End-to-End Feature Selection Method for Neural Networks

> [!IMPORTANT]
> I completely rewrote the algorithm from TensorFlow to PyTorch. The previous version is included in the tensorflow-2.9 branch, it is no longer maintained.
> This new version does not require the user to specify a model, it is completely automatic, although custom architectures can be used.

## Setup Instructions

### The python environment is included in the file requirements.txt.
Run the command:
 
    conda create --name e2efs --file ./requirements.txt

## HOW TO USE IT
Example included in example.py    

```python
## LOAD E2EFSSoft model
model = e2efs.E2EFSSoft(n_features_to_select=n_features_to_select)

## OPTIONAL: Load E2EFS Model
# model = e2efs.E2EFS(n_features_to_select=n_features_to_select)

## OPTIONAL: Load E2EFSRanking Model
# model = e2efs.E2EFSRanking()

## FIT THE SELECTION
model.fit(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=2000)

## FINETUNE THE MODEL
model.fine_tune(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=100)

## GET THE MODEL RESULTS
metrics = model.evaluate(test_data, test_label)
print(metrics)

## GET THE MASK
mask = model.get_mask()
print('MASK:', mask)

## GET THE RANKING
ranking = model.get_ranking()
print('RANKING:', ranking)
```
