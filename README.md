# Layout-aware Webpage Quality Assessment

## dependency

paddlepaddle-gpu==1.8.5

pgl==1.2.1

## data fromat

The data format of each line:

```
url_id \t json__data
```

where `json_data` is a json which has three keys:  node, edges, label


## Step 3: configuration file

All configuration can be found in `./src/config.yaml`:


## step 4: training

Run the following commands to train the model.

```
cd ./src
export CUDA_VISIBLE_DEVICES=0
python main.py --config config.yaml
```
