# Phase 1: Similarity Modeling 1 - Jump Detection

For the first Part of Similarity Modelling we started out by extracting the audio tracks from the videos to have the ability to analyse them seperatly. Our initial impression from looking at the files was, that the Jumps are rather visible when viewing the audio files in spectrogram view in audacity. Our initial idea was to just detect the jumps by finding these peaks in the audio. Because we wanted to initially stay away from Fourier Tranformation Features we used Zero Crossing Rate and Root Mean Square as our features and tried to manually find treshholds that would fit the jumps in our data. 

We came up with a few treshhold that worked for at least a few examples but we weren't satisified with this approach, because it didn't generalize well and it didn't use any machine learning. As a next step we used low and conservative treshholds to gather a subset of possible candidates and then fed the subset into a SVM. But with just using the two attributes we couldn't linearly seperate the feature space so that solution fell apart.

Our next idea was to use FFT to detect high values like we saw in the spectogram. For our initial implementation we converted the audio signal into the frequenzy domain using 1.5 second windows, to match the groundtruth which had a second accuracy. We then summed all the value over the 4mhz (from our observation of the spektrogram). This resulted in pretty usefull values overall, but we had a few anomalies, the funniest of which was a point that was always detected as a jump and after listening to the audio we found out that it was just someone laughing really loud. We kind of felt trolled :D

After that we went ahead and tried different typ of spectral features in conjunction with Neural Networks. A big problem we had for a while was deciding on the types of Windowing we wanted to use and how to trim windows down to get relevant spectral components but also applying to the whole window we needed to represent for purposed of our ground truth. Another problem we had was in the way we picked test samples for our neural network to train. Using the whole audio resulted in weird issues where we had way to many negative samples to actually train a network. We first solved this by selecting the positive samples and random negative samples but this seemed to lead to either overfitting for one or the other group.


# Phase 2: Similarity Modeling 2 - Hand Detection

The hand detection was fairly straight forward. We used OpenCV using Python3.

At first we tried to implement a rope detection, since the hands are always near the rope using Probabilistic Hough Lines. Unfortunately this did not work as expected.

The next try was using the following scheme:

 * simple HSV color based thresholding
 * Applying a gaussian blur to the mask
 * Extract the mask
 * This returns a more precise skin color region
 * Apply the following morphological operations: MORPH_ELLIPSE and MORPH_CLOSE to fight noise in the input signal
 * Find all contours in the input image which is classified as skin
 * Filter only contours that have reasonable area (not too small, not too big)
 * For each contour we extract the following metrics:
	* Skin Ratio: Amount of skin inside the bounding rect relative to the total area. The skin is now detected using a neural network we trained based on a publicly available dataset ( https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation )
	* Edge Ratio: Since skin is quite smooth and do not expect many edges, we apply a canny operator on the extracted skin relative to the total area
 * Now we take the best match of the available contours (using fixed tresholds). If there is no contour matching, the thresholds are relaxed a few times. If nothing is found, now contour is returned.
 * Draw the bounding boxes of the best 2 matches as possible hands. Since there is some detection locality expected, the bounding boxes are averaged, in order to smooth out the result. 

## Improvments:
 * Try to incorporate the mountings
 * Make the thresholding adaptive
 
# Phase 3: Similarity Modeling 1 and 2: Jump and Winch Detection

Both tasks are solved with basically the same algorithm.

## Feature Extraction:

For each video file we extracted "sound snippets" of 2 seconds. The following snippets are chosen:
 * 2 seconds around the jumps (if exists)
 * 2 seconds around the winch usage (if exists)
 * 2 seconds of 6 random samples of "background noise", where no notable event happens.

To avoid overfitting the ratio of positive examples and negative examples was chosen to be not too unbalanced (to avoid overfitting). Additionally, even though we could use a 3 second snippet time, we actively chose to use 2 seconds, to avoid
to get a "fuzzy" ground truth. Since a jump and winch event is a rather discrete event and if we considered audio as a jump, which was clearly not a jump we feared that the accuracy might drop.

For each of these samples we calculated the following features:
 * Absolute Short-Time Fourier Transform
 * Mean and standard deviation of Mel Frequency Cepstral Coefficients
 * Mean and standard deviation of the Chromagram based on the STFT
 * Mean and standard deviation of the Spectral Contrast
 * Mean and standard deviation of the Tonal Centroid Features
 * Mean and standard deviation of the Zero Crossing Rate
 * Mean and standard deviation of the Root-Mean-Square Energy
 * Mean and standard deviation of the normalized onset strength

Basically we shoved all the features we could get from the snippets and hoped that our neural network would generalize well on these. Which it luckily did. Since we were using the mean and standard deviation for many features it was important that our snippets contain as much "jump audio" as possible.

This added to a total number of 435 training samples. Each sample had a classification if it is a jump/winch usage or not.

For testing we used the same architecture, but instead of extracting of 6 random noise samples, we chose to extract 1000 noise samples. Since we needed to have a reasonable approximation of the other sounds that occur in the video.
We designated 10 videos for the testing set and the rest for the training set. Since the test data is quite sparse for positive cases, we could not afford to put more to the side.

## Training

We used Tensorflow in Python3 as our machine learning toolkit. We are using 2 hidden layer with 280 neurons on layer 1 and 300 neurons on layer 2 and trained using classical gradient descents with a learning rate of 0.01 and 5000 epochs. Since the training set was rather small we did not need any batching.

## Results

For the Jump Detection we get the following results:

Test Sample Size: 14030
Precision: 1.0
Recall: 0.99
F-Score: 0.995

For the Winch Detection we get the following results:

Test Sample Size: 14030
Precision: 1.0
Recall: 0.997
F-Score: 0.998

Problem with the method of our validation is, that each "sample" is weighted the same. For example around winch/jump usage sometimes the classification is wrong, but since this is just a very small fraction of the total number of samples, it seems to be "overshadowed" by the cases where classification is correct.

## Improvements

 * Try different parameters (especially the number of background noise samples of 6 was chosen rather arbitrary, maybe more could improve the quality even more)
 * Use more "interesting" points for training. Especially the audio surrounding sounds is sometimes classified incorrectly. Incorporating these also as events could increase accuracy. On the other hand the ground truth is very noisy, a jump is actually not a point in time, but rather a time period (even though it is very short). The problem arises with this: Is the annotated time the beginning of the jump? Middle of the jump? End of the jump? The more we rely on this fact the fuzzier the ground truth may become. Since "false negatives" might sneak in, were we classify parts which are clearly not a jump as a jump.
 * Use deep learning approaches

# Phase 4: Making it visible 

Wiring it all together. Saving and restoring a TensorFlow model is not as trivial as it might sound - clashes in variable names inside the graph are to be expected -, which took quite some time to find the right fix for. The Hand Detection displays whether a jump/winch usage is being detected at the current time. Which also needed to be programmed.
Finally an export was written which captured the interesting time periods (around jumps, winch usage etc.) and saved them as video files.

\*.jump{0|1}.avi contain the jumps
\*.winch{0|1}.avi contain the winch usages
\*.empty.avi contain the first few seconds, if there is no jump/winch usage.

