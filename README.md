# detect_and_classify_football_player
<p>in this project get video from 3 view of football pitch and detect at first player and then map them to football pitch 2d image and then use convolutional neural network to detect the team of them base on color and cropped image of them <p>
<h4 style="font-size:300px;"><b>classification part</b></h4>
<p>in this part at first we should have image of cropped football players image for this we use code <b>.py</b> for get frame of video in each <b>n second</b> and save them in folder <b>frames</b> and then use <b>program to crop player image</b> this croption program give us xml of info about where is player and size and location of their anchor box and save them in folder <b>xmls</b> and after that we use code <b>classify_football_player_with_svm_classifier.ipynb</b> to classification player with svm classifier and <b>classify_football_player_with_cnn_classifier.ipynb</b> to classification player with convolutional neural network that more explanation about code can be find in it ipynb file with codes</p>
<p>at end of the code of cnn classification have code that get input video by name of <b>"output.mp4"</b> that has been compiled before that is short time of complete and get that and map player location with color corresponding with thier tshirt color to 2d map and create video with connect that 2d map</p>
<h3>colab links</h3>
<p>https://colab.research.google.com/drive/1x_N9hzJmNzo15qhvWLelGmN3M9n_gpBK?usp=sharing</p>
<p>https://colab.research.google.com/drive/16EpDoJWTXCspbgkTSfihCVcDdriiKnyJ?usp=sharing</p>
