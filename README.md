# detect_and_classify_football_player
<p>in this project get video from 3 view of football pitch and detect at first player and then map them to football pitch 2d image and then use convolutional neural network to detect the team of them base on color and cropped image of them <p>
<h4><b>the raw database</b></h4>
<p> in this link we could access to 3 camera that take picture from 3 diffrent view</p>
<p>https://datasets.simula.no/alfheim/2013-11-03/First%20Half/</p>
<p>https://datasets.simula.no/alfheim/2013-11-03/Second%20Half/</p>
<p>https://datasets.simula.no/alfheim/2013-11-03/First%20Half/1/</p>
<p>https://datasets.simula.no/alfheim/2013-11-03/Second%20Half/1/</p>

<h4><b>How to generate longer videos:</b></h4>
<p>the videos are generated into 3-second individual video clips in pure H.264. To make longer videos into one file, it should be possible to concatenate the files directly using for example cat. However, if your player also needs a container, the easiest is perhaps to use ffmpeg and its concatenation option (-f concat). So, if you for example want to generate an MPEG4 file, in 1080p with a rate of 1 Mbps from minute 9 to minute 10 in the Tottenham game, you first generate a file (e.g., "files.txt") of the segments you want to include:

file '0180_2013-11-28 19:12:54.470740000.h264'
file '0181_2013-11-28 19:12:57.470788000.h264'
...
file '0198_2013-11-28 19:13:48.431859000.h264'
file '0199_2013-11-28 19:13:51.431950000.h264'   

Then, you for example run the below line to generate a new video to be written into the output.mp4 file:

ffmpeg -safe 0 -f concat -i files.txt output.mp4</p>

<h4><b>detect and generate spotted 2d map</b></h4>
<p>In this part, our primary focus is to detect everyone (including the referee) in the
scene. We also want to generate a 2D map of the field.
i assume that all players are standing straight on the ground (ignore jumps and slide tackles)</p>

<h4><b>Simultaneous picturing map to 2d image</b></h4>
<p>in this part we get 3 camera movie get part of it by program <b>film cutting program</b> and then detect players and refree Simultaneous in 3 movie and map them in to the 2d map and make video from them by connect map of each frame </p>
<p>the important point in this code is if the players or refree be in 3 or 2 cammera only one time should be spotted in 2d image</p>
<p>code of this part are in folder by name "Simultaneous picturing map to 2d image"

<h4><b>object detection</b></h4>
<p>object detection in frames and flow players with id till get out of cammer filed of view
for do this we use code When the players get close to each other, they don't get lost as much as possible. For this, it is better to use Tracker.
In the codes of this folder, we follow the players between different frames. We consider a desired id for each player and identify the player with the same id in different frames using trackers. (Until the player leaves the frame or is lost for any other reason).
code of this part are in folder of <b>"tracking players"</b>
</p>
<h4><b>classification part</b></h4>
<p>In this section, we assume we have the positions of the players. Now, we want to classify them. Keep in mind that the 2D map colors do not have to match the actual team’s color. The only goal is to separate players.
every code in this part of explanation are in folder of <b>"classification"</b>
in this part at first we should have image of cropped football players image for this we use code <b>.py</b> for get frame of video in each <b>n second</b> and save them in folder <b>frames</b> and then use <b><a href="https://github.com/tzutalin/labelImg">labelimg</a></b> this croption program give us xml of info about where is player and size and location of their anchor box and save them in folder <b>xmls</b> and after that we use code <b>classify_football_player_with_svm_classifier.ipynb</b> to classification player with svm classifier and <b>classify_football_player_with_cnn_classifier.ipynb</b> to classification player with convolutional neural network that more explanation about code can be find in it ipynb file with codes</p>
<p>at end of the code of cnn classification have code that get input video by name of <b>"output.mp4"</b> that has been compiled before that is short time of complete and get that and map player location with color corresponding with thier tshirt color to 2d map and create video with connect that 2d map</p>
<h3>colab links</h3>
<p>https://colab.research.google.com/drive/1x_N9hzJmNzo15qhvWLelGmN3M9n_gpBK?usp=sharing</p>
<p>https://colab.research.google.com/drive/16EpDoJWTXCspbgkTSfihCVcDdriiKnyJ?usp=sharing</p>
