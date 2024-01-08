### Wind Power Prediction

### How to run?
```
python cnn_lstm_autoreg_v8.py (best model with autoregression)
python cnn_lstm_v4.py (best model without autoregression)
```

## Summary of Runs and best eval scores
**NOTE: Only few important runs are kept for comparison**

### Run ID - Description
- 20240102235836 - Linear layer
- 20240103010304 - Transfomer + Linear + (diff heads + diff layers exp)
- 20240103145103 - (cnn_lstm_v4.py Script) CNN + LSTM + dropout - **eval loss = 0.33 (249677 params)**
- 20240103155901 - CNN + LSTM with temporal attention 
- 20240103172913 - Resnet + LSTM 
- 20240103173602 - Resnet + LSTM + temporal attention
- 20240104003229 - (cnn_lstm_autoreg_v8.py Script) Autoregression/LSTM + CNN + LSTM - **eval_loss: 0.1095 - params (467917 params)**

**General notes:**
- Resnet block did not gave a significant boost to accuracy with parameter increase, so I eneded up using normal convolution block.
- Temporal attention was used to provide different imprtance to each time step for the impact on current wind power, but I didn't see any perf boost, maybe it will help in higher data regimes!  

### Other potential ideas to try if the data is constrained:
- More CNN + Transformer experiements
- Data augmentation 
    - Temporal: Vary window size of input, temporal shift of 1/2 hrs
    - Spatial: Apply rotation/translation transforms to 14*14*3 features
- If location of  wind farms is given we can try Graph Convolutions on previous day's wind power values


**PS:** We can discuss more on experiments that I tried and potential directions if more time/data is given! 