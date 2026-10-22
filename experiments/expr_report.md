Classifier: Linear(in_features=1024, out_features=4, bias=True)

Device: cuda

Train samples: 5600 | Test samples: 1600

# RadImageNet

## Classifier

Trainable params: 4100
| epoch|loss |time elapsed|
| -|- |-|
Epoch 01/10 | Loss: 1.2468 | ET: 29.03s
Epoch 05/10 | Loss: 0.8952 | ET: 28.93s
Epoch 10/10 | Loss: 0.7436 | ET: 28.77s

10 epoch done
Total ET: 291.13s (4.85 mins)
avg ET per epoch: 29.11s

## tes 1 block

Trainable params: 2164228
| epoch|loss |acc |time elapsed |
|- |- |- |- |
Epoch 1 | Loss: 1.3694 | Accuracy: 0.4216 | Training time: 27.98s
Epoch 2 | Loss: 1.2535 | Accuracy: 0.7532 | Training time: 27.81s
Epoch 3 | Loss: 0.9798 | Accuracy: 0.7966 | Training time: 27.34s
Epoch 4 | Loss: 0.7040 | Accuracy: 0.8313 | Training time: 27.67s
Epoch 5 | Loss: 0.5326 | Accuracy: 0.8555 | Training time: 27.63s
Epoch 6 | Loss: 0.4333 | Accuracy: 0.8700 | Training time: 27.56s
Epoch 7 | Loss: 0.3741 | Accuracy: 0.8827 | Training time: 27.50s
Epoch 8 | Loss: 0.3447 | Accuracy: 0.8862 | Training time: 27.65s
Epoch 9 | Loss: 0.3141 | Accuracy: 0.8973 | Training time: 27.41s
Epoch 10 | Loss: 0.2856 | Accuracy: 0.9048 | Training time: 27.59s

## tes 2 block

Trainable params: 5528324
| epoch|loss |acc |time elapsed |
|- |- |- |- |
Epoch 1 | Loss: 1.3566 | Accuracy: 0.3409 | Training time: 31.71s
Epoch 2 | Loss: 1.1646 | Accuracy: 0.7782 | Training time: 32.07s
Epoch 3 | Loss: 0.8040 | Accuracy: 0.8266 | Training time: 32.05s
Epoch 4 | Loss: 0.5284 | Accuracy: 0.8668 | Training time: 32.16s
Epoch 5 | Loss: 0.3957 | Accuracy: 0.8850 | Training time: 32.02s
Epoch 6 | Loss: 0.3328 | Accuracy: 0.8954 | Training time: 32.20s
Epoch 7 | Loss: 0.2907 | Accuracy: 0.9082 | Training time: 32.32s
Epoch 8 | Loss: 0.2551 | Accuracy: 0.9166 | Training time: 32.32s
Epoch 9 | Loss: 0.2315 | Accuracy: 0.9202 | Training time: 33.88s
Epoch 10 | Loss: 0.2045 | Accuracy: 0.9302 | Training time: 32.15s

## full ft

Trainable parameters: 6957956
| epoch|loss |acc |time elapsed |
|- |- |- |- |
Epoch 1/10 | Loss: 1.3443 | Accuracy: 0.5641 | Training time: 58.43s
Epoch 2/10 | Loss: 1.0800 | Accuracy: 0.7575 | Training time: 58.50s
Epoch 3/10 | Loss: 0.7178 | Accuracy: 0.8370 | Training time: 58.53s
Epoch 4/10 | Loss: 0.4943 | Accuracy: 0.8666 | Training time: 58.89s
Epoch 5/10 | Loss: 0.3722 | Accuracy: 0.8884 | Training time: 60.19s
Epoch 6/10 | Loss: 0.3087 | Accuracy: 0.9039 | Training time: 59.27s
Epoch 7/10 | Loss: 0.2625 | Accuracy: 0.9216 | Training time: 58.66s
Epoch 8/10 | Loss: 0.2320 | Accuracy: 0.9280 | Training time: 58.90s
Epoch 9/10 | Loss: 0.1930 | Accuracy: 0.9407 | Training time: 58.67s
Epoch 10/10 | Loss: 0.1646 | Accuracy: 0.9495 | Training time: 58.65s


# ImageNet

## Classifier

Trainable params: 4100

training 10 epoch
| epoch|loss |time elapsed|
| -|- |-|
Epoch 01/10 | Loss: 0.6516 | ET: 32.20s
Epoch 05/10 | Loss: 0.2846 | ET: 34.58s
Epoch 10/10 | Loss: 0.2386 | ET: 34.66s

10 epoch done
Total ET: 309.39s (5.16 mins)
avg ET per epoch: 30.94s

## Tes 1 block

Trainable params: 2164228
| epoch|loss |acc |time elapsed |
|- |- |- |- |
Epoch 1 | Loss: 0.4016 | Accuracy: 0.8641 | Training time: 32.34s
Epoch 2 | Loss: 0.1352 | Accuracy: 0.9595 | Training time: 31.11s
Epoch 3 | Loss: 0.0580 | Accuracy: 0.9861 | Training time: 34.80s
Epoch 4 | Loss: 0.0307 | Accuracy: 0.9943 | Training time: 28.61s
Epoch 5 | Loss: 0.0199 | Accuracy: 0.9952 | Training time: 28.08s
Epoch 6 | Loss: 0.0227 | Accuracy: 0.9946 | Training time: 28.45s
Epoch 7 | Loss: 0.0125 | Accuracy: 0.9973 | Training time: 28.01s
Epoch 8 | Loss: 0.0105 | Accuracy: 0.9986 | Training time: 27.64s
Epoch 9 | Loss: 0.0079 | Accuracy: 0.9989 | Training time: 27.96s
Epoch 10 | Loss: 0.0045 | Accuracy: 0.9995 | Training time: 28.62s

## Tes 2 block

Device: cuda

Trainable params: 5528324
| epoch|loss |acc |time elapsed |
|- |- |- |- |
Epoch 1 | Loss: 0.2723 | Accuracy: 0.9125 | Training time: 37.54s
Epoch 2 | Loss: 0.0503 | Accuracy: 0.9873 | Training time: 38.17s
Epoch 3 | Loss: 0.0236 | Accuracy: 0.9945 | Training time: 36.46s
Epoch 4 | Loss: 0.0157 | Accuracy: 0.9973 | Training time: 37.32s
Epoch 5 | Loss: 0.0127 | Accuracy: 0.9975 | Training time: 36.73s
Epoch 6 | Loss: 0.0140 | Accuracy: 0.9961 | Training time: 36.93s
Epoch 7 | Loss: 0.0052 | Accuracy: 0.9996 | Training time: 37.14s
Epoch 8 | Loss: 0.0023 | Accuracy: 1.0000 | Training time: 37.03s
Epoch 9 | Loss: 0.0015 | Accuracy: 1.0000 | Training time: 36.81s
Epoch 10 | Loss: 0.0133 | Accuracy: 0.9964 | Training time: 36.70s

## full ft

Trainable parameters: 6957956
| epoch|loss |acc |time elapsed |
|- |- |- |- |
Epoch 1/10 | Loss: 0.2530 | Accuracy: 0.9155 | Training time: 70.51s
Epoch 2/10 | Loss: 0.0555 | Accuracy: 0.9868 | Training time: 68.91s
Epoch 3/10 | Loss: 0.0197 | Accuracy: 0.9957 | Training time: 69.14s
Epoch 4/10 | Loss: 0.0212 | Accuracy: 0.9930 | Training time: 68.81s
Epoch 5/10 | Loss: 0.0139 | Accuracy: 0.9968 | Training time: 68.45s
Epoch 6/10 | Loss: 0.0125 | Accuracy: 0.9973 | Training time: 68.94s
Epoch 7/10 | Loss: 0.0057 | Accuracy: 0.9988 | Training time: 69.22s
Epoch 8/10 | Loss: 0.0090 | Accuracy: 0.9979 | Training time: 69.49s
Epoch 9/10 | Loss: 0.0081 | Accuracy: 0.9977 | Training time: 69.13s
Epoch 10/10 | Loss: 0.0231 | Accuracy: 0.9927 | Training time: 68.35s
