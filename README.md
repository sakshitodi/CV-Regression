# CV-Regression
A deep learning pipeline for Financial Predictions

In this project, I explore different existing methods of financial predictions (stock price), and show how converting time series data to images can out-perform existing methods.


## Simple CNN based Predictions

<img width="1381" alt="image" src="https://github.com/sakshitodi/CV-Regression/assets/95418108/b9eae3ea-6f8b-433f-b244-12a8dcd03303">

A non-sequential model like CNN fails to capture the trend and performs poorly.

## LSTM
<img width="1372" alt="image" src="https://github.com/sakshitodi/CV-Regression/assets/95418108/282d5ce6-a8d0-4012-b773-f4afa3520955">

Here, it captures the trend but we can do better.

## GADF Encoding

This a mathematical approach to capture the time-series trends based on cosine similarity and then convert it into images, which is more useful since more information can be extracted from an image, rather than just a column.
<img width="410" alt="image" src="https://github.com/sakshitodi/CV-Regression/assets/95418108/590ba89f-09c3-41b5-856b-9f63fb9781da">

## Training an LSTM on GADF generated images
I finally generated such images for each time stamp using a sliding window method, then captured its embeddings using a RESNET and finally added an LSTM layer to capture sequential patterns.

<img width="1370" alt="image" src="https://github.com/sakshitodi/CV-Regression/assets/95418108/5c4798ea-7fb1-4883-8a3e-0763ed879793">

As we see, this makes better predictions than before, and captures trend more effectively. 
## Generating using a DL model
Now, instead of using GADF to generate images, I kept a sequential pipeline which encodes time-series to images and then trains an LSTM for predictions.
<img width="1365" alt="image" src="https://github.com/sakshitodi/CV-Regression/assets/95418108/be7af292-d358-45b5-a7c1-db5d6615242c">

Upon a more careful evaluation, we see that this method has even less Mean Squared error for unseen data and therefore performs even better.
