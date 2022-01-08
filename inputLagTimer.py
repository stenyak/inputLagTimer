#!/usr/bin/env python
###############################################################################
# Copyright 2021 Bruno Gonzalez Campo <stenyak@stenyak.com> (@stenyak)        #
# Distributed under MIT license (see license.txt)                             #
###############################################################################

import os, sys, math, json
import cv2 as cv
import numpy as np

maxLatency = 0.7 # seconds
version = "1.1"

###############################################################################
#### helper classes ###########################################################

# each single motion detection
class MotionDetection:
  def __init__(self, kind, score, scoreThreshold, duration):
    self.kind = kind # "in" or "out" for motion detected inside either rectangle
    self.score = score
    self.scoreThreshold = scoreThreshold
    self.duration = duration
# each measurement of photon-to-photon latency
class Latency:
  def __init__(self):
    self.inDetectedFirst = 0
    self.inDetectedLast = 0
    self.outDetectedFirst = 0
    self.outDetectedLast = 0
    self.lastFrame = 0
    self.motionDetections = []
    self.startTime = 0
  def secs(self, spf):
    if self.outDetectedFirst:
      return spf*(self.outDetectedFirst - self.inDetectedFirst)
    else:
      return self.total(spf)
  def total(self, spf):
    return spf*(self.lastFrame - self.inDetectedFirst)


###############################################################################
#### user interface drawers ###################################################

# helper vars
green    = [ 20, 162,  73]
red      = [111, 111, 200]
blue     = [255, 180,   0]
purple   = [255,   0, 210]
yellow   = [  0, 210, 210]
white    = [255, 255, 255]
black    = [  0,   0,   0]
grey     = [200, 200, 200]
darkgrey = [ 70,  70,  70]
inputColor = blue
outputColor= purple
lineHeight = 16

def getLineY(line):
  return lineHeight*(line+1)

# draws text on screen. vertical coordinates can be provided in pixels, or in lines of text (at default font size)
def drawText(frame, text, x, y=None, line=None, fontScale=0.4, thickness=0, lineType=cv.LINE_AA, color=white, colorbg=black):
  font = cv.FONT_HERSHEY_SIMPLEX
  bottomLeftCornerOfText = (x, y if y else getLineY(line))
  if colorbg is not None:
    cv.putText(frame, text, bottomLeftCornerOfText, font, fontScale, colorbg, thickness+2, lineType)
  cv.putText(frame, text, bottomLeftCornerOfText, font, fontScale, color, thickness, lineType)

# draws a rectangle. thickness of -1 fills the rectangle. optional scale can be applied to the rectangle
def drawRect(frame, rect, color, thickness=-1, scaleX=1, scaleY=1):
  if rect is None:
    return frame
  x,y,w,h = rect
  x *= scaleX
  y *= scaleY
  w *= scaleX
  h *= scaleY
  start = (round(x), round(y))
  end   = (round(x + w), round(y + h))
  return cv.rectangle(frame, start, end, color, thickness)

def drawHeaderText(frame, currTime, videoEnded, line):
  drawText(frame, "{:01}:{:06.3f} (s)elect rectangles, (p)ause, (r)estart, (esc)exit, (a)dvanced".format(int(currTime/1000/60), currTime/1000 % 60,), 5, line=line)
  line += 1
  if videoEnded:
    drawText(frame, " ------ Reached end of video ------", 5, line=line, color=red, colorbg=white)
    line += 1
  return line

# indicator of input and output motion scores, including label, last detected peak, and motion scoreThreshold
def drawBar(frame, x, y, width, maxValue, scoreThreshold, score, scorePeak, scorePeakValid, color):
  border = 1
  height = lineHeight-5
  def scale(v):
    return (width-2*border)*min(v, maxValue) / maxValue
  h = height;              w = width
  frame = drawRect(frame, [x                        , y-0.5*h, w, h], darkgrey)
  frame = drawRect(frame, [x                        , y-0.5*h, w, h], black, 1)
  h = height-2*border; w = scale(score)
  frame = drawRect(frame, [x+border                 , y-0.5*h, w, h], color)
  if scorePeak:
    h = height-2*border;     w = 1
    frame = drawRect(frame, [x-border+w+scale(scorePeak)     , y-0.5*h, w, h], color)
  if scorePeakValid:
    h = height-2*border;     w = 3
    frame = drawRect(frame, [x-border+w+scale(scorePeakValid)     , y-0.5*h, w, h], color)
  h = height-2*border;     w = 1
  frame = drawRect(frame, [x-border+w+scale(scoreThreshold), y-0.5*h, w, h], white)
  return frame

# displays a Latency measurement, including all its motion events, the cooldown period, etc
def drawLatency(frame, advanced, startTime, spf, n, yStart, secs, total, colorframe, colorbg, colortext, text, motionDetections, cooldown=None):
  x = 5
  h = 16
  ms = round(1000*secs)
  totalms = round(1000*total)
  y = yStart + n*h
  if advanced:
    drawText(frame, "{:01}:{:06.3f}".format(int(startTime/1000/60), startTime/1000 % 60), x, y=y+h-4, fontScale=0.4, thickness=1, color=colortext, colorbg=colorbg)
    x = 5+70
  if cooldown:
    cooldownms = round(1000*cooldown)
    frame=drawRect(frame, [x,   y,   cooldownms+totalms+2, h  ], colorframe)
    frame=drawRect(frame, [x+1, y+1, cooldownms+totalms  , h-2], darkgrey)
  if cooldown or advanced:
    frame = drawRect(frame, [x,   y,      totalms+2, h  ], colorframe)
    frame = drawRect(frame, [x+1, y+1,    totalms  , h-2], grey)
  frame = drawRect(frame, [x,   y,           ms+2, h  ], colorframe)
  frame = drawRect(frame, [x+1, y+1,         ms  , h-2], colorbg)
  if advanced:
    for motionDetection in motionDetections:
      scoreMax = motionDetection.scoreThreshold*5
      scoreNorm = min(1, (motionDetection.score-motionDetection.scoreThreshold) / scoreMax)
      eventPixels = max(2, 1000*spf*scoreNorm)
      if motionDetection.kind == "in":
        frame = drawRect(frame, [x+1 + 1000*motionDetection.duration, y+h-1, eventPixels, -2], inputColor)
      else:
        frame = drawRect(frame, [x+1 + 1000*motionDetection.duration, y+1  , eventPixels,  2], outputColor)
  drawText(frame, text, x+1, y=y+h-4, fontScale=0.4, thickness=1, color=colortext, colorbg=None)
  return frame


###############################################################################
#### user interface ###########################################################

# helper vars
windowName = "InputLagTimer v{}".format(version)
advanced = False

# blocks the UI, asking the user to draw a rectangle with the mouse
def requestRectangle(windowName, frame, winWidth, winHeight):
  fromCenter = False
  showCrossHair = False
  r = cv.selectROI(windowName, frame, showCrossHair, fromCenter)
  if r[2] == 0 or r[3] == 0:
    r = None
  else:
    r = [r[0]/winWidth, r[1]/winHeight, r[2]/winWidth, r[3]/winHeight]
  return r

# blocks the UI, asking the user to choose the Input and the Output rectangles
def selectRectangles(frame, winWidth, winHeight, windowName, config, dirtyConfig):
  # input rectangle selection
  frameRender = frame.copy()
  frameRender = cv.resize(frameRender, (winWidth, winHeight))
  frameRender = drawRect(frameRender, config['inputRectangle'] ,  inputColor, 5, winWidth, winHeight)
  frameRender = drawRect(frameRender, config['outputRectangle'], outputColor, 1, winWidth, winHeight)
  line = 0
  drawText(frameRender, "Draw a rectangle where input motion is expected, then press SPACE key", 5, line=line, color=inputColor, colorbg=white)
  line += 1
  drawText(frameRender, "(to keep current rectangle, press SPACE without drawing a new one)", 5, line=line, color=inputColor, colorbg=white)
  line += 1
  drawText(frameRender, "E.g. a key on your keyboard, a gamepad stick, a proximity sensor...", 5, line=line)
  line += 1
  inputRectangle = requestRectangle(windowName, frameRender, winWidth, winHeight)
  if inputRectangle is not None: # if user didn't select rectangle, keep previous one
    dirtyConfig = True
    config['inputRectangle'] = inputRectangle

  # output rectangle selection
  frameRender = frame.copy()
  frameRender = cv.resize(frameRender, (winWidth, winHeight))
  frameRender = drawRect(frameRender, config['inputRectangle'] ,  inputColor, 1, winWidth, winHeight)
  frameRender = drawRect(frameRender, config['outputRectangle'], outputColor, 5, winWidth, winHeight)
  line = 0
  drawText(frameRender, "Draw a rectangle where output motion is expected, then press SPACE key", 5, line=line, color=outputColor)
  line += 1
  drawText(frameRender, "(to keep current rectangle, press SPACE without drawing a new one)", 5, line=line, color=outputColor)
  line += 1
  drawText(frameRender, "E.g. the middle of a screen, a steering wheel actuated with force-feedback, an oscilloscope...", 5, line=line)
  line += 1
  outputRectangle = requestRectangle(windowName, frameRender, winWidth, winHeight)
  if outputRectangle is not None: # if user didn't select rectangle, keep previous one
    dirtyConfig = True
    config['outputRectangle'] = outputRectangle

  return dirtyConfig

# respond to whatever the user has pressed on the keyboard
def processKeypress(key, config, retry, paused, pauseOnce, dirtyConfig, selectRectanglesRequested, cap):
  breakRequested = False
  webcamNextRequested = False
  if key == 27: # quit
    breakRequested = True
  if key == ord('r'): # restart
    retry = True
    breakRequested = True
  if key == ord('a'): # advanced
    global advanced
    advanced = not advanced
    pauseOnce = True
  if key == ord('q'): # pause
    currQuality = qualities.index(config['quality'])
    config['quality'] = qualities[(currQuality+1) % len(qualities)]
    dirtyConfig = True
    retry = True
    breakRequested = True
  if key == ord('p'): # pause
    paused = not paused
  if key == ord('w'): # next webcam
    breakRequested = True
    webcamNextRequested = True
  if key == ord('c'): # webcam config
    cap.set(cv.CAP_PROP_SETTINGS, 1)
  if key == ord('m'): # mode
    currMode = modes.index(config['detectMode'])
    config['detectMode'] = modes[(currMode+1) % len(modes)]
    dirtyConfig = True
    pauseOnce = True
  if key == ord('s'): # select rectangles
    selectRectanglesRequested = True
    dirtyConfig = True
  elif key == ord('n'):
    pauseOnce = True
  elif key == ord('+'):
    config['edgeThreshold'] *= 1.2
    dirtyConfig = True
    pauseOnce = True
  elif key == ord('-'):
    config['edgeThreshold'] /= 1.2
    dirtyConfig = True
    pauseOnce = True
  elif key == ord('1'):
    config['inThreshold'] /= 1.2
    dirtyConfig = True
    pauseOnce = True
  elif key == ord('2'):
    config['inThreshold'] *= 1.2
    dirtyConfig = True
    pauseOnce = True
  elif key == ord('3'):
    config['outThreshold'] /= 1.2
    dirtyConfig = True
    pauseOnce = True
  elif key == ord('4'):
    config['outThreshold'] *= 1.2
    dirtyConfig = True
    pauseOnce = True
  return breakRequested, retry, paused, pauseOnce, dirtyConfig, selectRectanglesRequested, webcamNextRequested


###############################################################################
#### image processing #########################################################

# helper vars
modes = [ "abs", "edges", "abs+edges" ]

# writes a sub-rectangle into the provided image
def drawSubImage(frame, subImage, multiply, rect, scaleX, scaleY):
  if rect is None:
    return
  x,y,w,h = rect
  frame[round(scaleY*y):round(scaleY*y)+subImage.shape[0], round(scaleX*x):round(scaleX*x)+subImage.shape[1]] = multiply*subImage

# returns a sub-rectangle out of the provided image
def readSubImage(frame, rect, scaleX, scaleY):
  if rect is None:
    return None
  x,y,w,h = rect
  return frame[round(scaleY*y):round(scaleY*(y+h)), round(scaleX*x):round(scaleX*(x+w))]

def computeImageEdges(frame, config):
  if frame is None:
    return None
  frameBlur = cv.blur(frame, (3,3))
  kernelSize = 3
  thresholdMin = config['edgeThreshold'] - 1
  thresholdMax = min(0.1, thresholdMin*1.5)
  edges = cv.Canny(frameBlur, thresholdMin, thresholdMax, kernelSize)
  mask = edges != 0
  return frame * (mask[:,:,None].astype(frame.dtype))

# checks if the diff represents a meaningful amount of movement, based on analysis of historical movements
# e.g. do not mistake minor image vibrations for "motion" if the video tends to be shakey, etc
def computeMotionDetectionScore(nFrame, historySize, diff, values, scoreThreshold):
  motionDetected = False
  score = 0
  valueAvg = 0

  if diff is not None:
    # compute amount of change in pixels
    value = diff.sum()

    if values:
      # normalize current and historical values against average
      # this makes calculations independent of e.g. how big are the selected rectangles of image
      valueAvg = sum(values) / len(values)
      valueNorm = value / valueAvg if valueAvg > 0 else 0
      valuesNorm = [ v / valueAvg if valueAvg > 0 else 0 for v in values ]

      # calculate historical average absolute-deviation. this is an indicator of image stability
      #  - e.g. a really noisy video might result in 1.2 (which means +/- 120% around the average)
      #  - while very stable imagery might result in 0.1 (which means +/- 10% around the average)
      valueNormAvg = sum([abs(1-v) for v in valuesNorm]) / len(valuesNorm)

      # let's now compute a 'score', which represents the amount of motion detected
      score = max(0, valueNorm - valueNormAvg)

      # knowing the stability of the image (valueNormAvg) allows us to guesstimate how to tweak the 'score' appropriately:
      #  - motion in a very stable image is detected by comparing the score to small thresholds
      #  - motion in a very noisy image is detected by comparing the score to larger thresholds
      # instead of scaling the threshold, we will scale the score (so, inversely, hence the 1/xxx division below)
      # (this way the UI can display a constant threshold value to the user, should they need to tweak it)
      # the we proceed to scale the score accordingly. based on tests, a ramp with these parameters yields the best results
      base = 0.2 # perfectly stable imagery will decrease threshold down to only 0.2
      ramp = 1.0 # every unit of 'image-instability' increases the threshold compensation-factor by 1.0
      compensation = base + ramp*valueNormAvg
      score *= 1/compensation

      # once we have a usable score from enough historic data, check if the amount of motion surpassed the desired threshold
      if len(values) >= historySize and score > scoreThreshold:
        motionDetected = True

    # add values to historic records, giving less weight to values corresponding to trigger events (to avoid skewing the baseline data)
    if value > valueAvg:
      if motionDetected:
        values.append(valueAvg + 0.1*(value-valueAvg)) # increase very slowly when the value was a trigger
      else:
        values.append(valueAvg + 0.5*(value-valueAvg)) # increase      slowly when the value is greater than avg
    else:
      values.append(value)  # decrease at normal speed when the value is below avg
    while(len(values) > historySize): values.pop(0) # limit historic size

  return motionDetected, score

# choose a frame-comparison method, and pass that diff through the motion detection system
def detectMotion(prevFrame, currFrame, nFrame, historySize, rect, frameWidth, frameHeight, config, scores, scoreThreshold):
  prev = readSubImage(prevFrame, rect, frameWidth, frameHeight)
  curr = readSubImage(currFrame, rect, frameWidth, frameHeight)
  if config['detectMode'] == "abs+edges":
    prevEdges = computeImageEdges(prev, config)
    currEdges = computeImageEdges(curr, config)
    diffEdges = cv.absdiff(prevEdges, currEdges)
    diffAbs = cv.absdiff(prev, curr)
    diff = cv.add(diffAbs, diffEdges)
  elif config['detectMode'] == "edges":
    prevEdges = computeImageEdges(prev, config)
    currEdges = computeImageEdges(curr, config)
    diff = cv.absdiff(prevEdges, currEdges)
  elif config['detectMode'] == "abs":
    diff = cv.absdiff(prev, curr)
  motionDetected, score = computeMotionDetectionScore(nFrame, historySize, diff, scores, scoreThreshold)
  return motionDetected, score, diff


###############################################################################
#### configuration ############################################################

# helper vars
qualities = [ "720p", "1080p", "VGA", "SVGA" ]

# reads the per-video configuration file from disk (e.g. myFile.mp4.cfg)
def loadConfig(videopath):
  try:
    with open("{}.cfg".format(videopath)) as f:
      config = json.load(f)
  except:
    config = {}

  #sanitize config
  if not 'quality'         in config: config['quality'        ] = qualities[0]
  if not 'inputRectangle'  in config: config['inputRectangle' ] = None
  if not 'outputRectangle' in config: config['outputRectangle'] = None
  if not 'edgeThreshold'   in config: config['edgeThreshold'  ] = 100
  if not 'inThreshold'     in config: config['inThreshold'    ] = 10
  if not 'outThreshold'    in config: config['outThreshold'   ] = 10
  if not 'detectMode'      in config: config['detectMode'     ] = modes[0]
  if not config['detectMode'] in modes:config['detectMode'    ] = modes[0]
  return config

# stores the currently used configuration to disk (e.g. myFile.mp4.cfg)
def saveConfig(videopath, config):
  # sanitize config
  config['edgeThreshold'] = max(config['edgeThreshold'], 1)
  config['inThreshold'] = max(config['inThreshold'], 0.25)
  config['outThreshold'] = max(config['outThreshold'], 0.25)

  # save to disk
  with open("{}.cfg".format(videopath), 'w') as f:
    json.dump(config, f)


###############################################################################
#### startup ##################################################################

def showCommandLineUsage():
  print("")
  print("Usage: {} VIDEO_SOURCE".format(sys.argv[0]))
  print("E.g.: {} file.mp4".format(sys.argv[0]))
  print("E.g.: {} 0".format(sys.argv[0]))

# parses the command line video string (string for file, number for webcam)
def getVideopath(webcamNextRequested):
  # i really hate how i've ended up handling argument parsing and the webcam cycling stuff, but can't be bothered to improve it. pull requests are welcome \o/
  if len(sys.argv) < 2:
    print("Error: missing argument, trying to open a webcam")
    showCommandLineUsage()
    webcamNextRequested = True
    sys.argv.append(-1) # begin with first webcam

  try:
    sys.argv[1] = int(sys.argv[1])
  except ValueError:
    # not a number
    if os.path.isfile(sys.argv[1]):
      if not webcamNextRequested:
        return sys.argv[1], webcamNextRequested
    else:
      print("Error: the provided path is not a number nor a file: {}".format(sys.argv[1]))
      showCommandLineUsage()
      return None, webcamNextRequested

  if webcamNextRequested:
    if type(sys.argv[1]) is int:
      sys.argv[1] += +1
      if sys.argv[1] > 15:
        print("Error: will not attempt to open any webcam beyond number #{}".format(sys.argv[1]))
        webcamNextRequested = False
        return None, webcamNextRequested
    else:
      sys.argv[1] = 0

  return sys.argv[1], webcamNextRequested

# start reading the video: use a string for file, or a number for webcam
def openVideopath(videopath, quality):
  # basic loading screen (since sometimes MSMF backend takes 20 seconds to open a webcam)
  frameRender = np.zeros((1000, 1000, 3), dtype = "uint8")
  frameRender[:] = darkgrey
  txt = "Accessing webcam #{}".format(videopath) if type(videopath) is int else "Opening file '{}'".format(videopath)
  print(txt, flush=True)
  drawText(frameRender, txt, 10, 100, fontScale=1.4, thickness=2, color=yellow)
  drawText(frameRender, "Please wait...", 10, 140, fontScale=1.4, thickness=2, color=yellow)
  cv.imshow(windowName, frameRender)

  # open camera
  if type(videopath) is int:
    # we have a webcam id number
    windowsHack = True
    if windowsHack and os.name == 'nt':
      cap = cv.VideoCapture(videopath, cv.CAP_DSHOW) # seems much faster than MSMF
      #cap = cv.VideoCapture(videopath, cv.CAP_MSMF)
      #cap = cv.VideoCapture(videopath, cv.CAP_WINRT)
      #cap = cv.VideoCapture(videopath, cv.CAP_VFW)
      #cap = cv.VideoCapture(videopath, cv.CAP_GSTREAMER)
      #cap = cv.VideoCapture(videopath, cv.CAP_FFMPEG)
      #cap = cv.VideoCapture(videopath, cv.CAP_OPENCV_MJPEG)
    else:
      cap = cv.VideoCapture(videopath)

    #cap.set(cv.CAP_PROP_FPS, 60) #let's at least try
    #cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    #cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'X264'))
    if quality == "1080p":
      cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    elif quality == "720p":
      cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    elif quality == "SVGA":
      cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
    elif quality == "VGA":
      cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
  else:
    # we have a filepath string
    cap = cv.VideoCapture(videopath)
  return cap


###############################################################################
#### main loop ################################################################

# processes a video from its beginning
# this function will be called each time you switch to a different webcam, or restart the playback of a video
def main(webcamNextRequested):
  print(" ===== Starting... ===== ")
  # loop control
  pauseOnce = False
  paused = False
  nFrame = -1
  videoEnded = False
  selectRectanglesRequested = False
  retry = False

  # find which video we will open
  videopath, webcamNextRequested = getVideopath(webcamNextRequested)
  if videopath is None:
    return retry, webcamNextRequested

  # go get the configuration for this video
  config = loadConfig(videopath)

  cv.namedWindow(windowName, cv.WINDOW_NORMAL)
  cap = openVideopath(videopath, config["quality"])

  if not cap.isOpened():
    if type(videopath) is int:
      print("Error: unable to open capture for: webcam #{}".format(videopath))
    else:
      print("Error: unable to open capture for file: {}".format(videopath))
    return retry, webcamNextRequested

  if type(videopath) is int:
    ret, newFrame = cap.read()
    if not ret:
      print("Error: unable to read images from webcam #{}".format(videopath))
      return retry, webcamNextRequested

  fps = cap.get(cv.CAP_PROP_FPS)
  if fps == 0:
    #return retry, webcamNextRequested
    defaultFps = 30
    print("Error: no FPS detected: {}. Defaulting to {}".format(fps, defaultFps))
    fps = defaultFps
  webcamNextRequested = False

  # statistics
  historySize = round(0.3 * fps)
  inScores = []
  outScores = []

  # latency detection
  inScore = 0
  outScore = 0
  inScorePeakValid = 0
  outScorePeakValid = 0
  inScorePeak = 0
  outScorePeak = 0
  cooldown = 0
  latency = Latency()
  latencies = []

  # aux data
  currTime = 0
  currFrame = None
  prevFrame = None
  spf = 1. / fps

  while cap.isOpened():
    # read a new image from the video if needed/possible
    dirtyConfig = False
    if not pauseOnce:
      ret, newFrame = cap.read()
      if ret:
        currTime = cap.get(cv.CAP_PROP_POS_MSEC)
        prevFrame = currFrame
        currFrame = newFrame
        nFrame += 1
      else:
        # the video has ended
        paused = True
        videoEnded = True
        cooldown = 0 # don't display a dangling cooldown, just finish it
        if currFrame is None:
          # couldn't retrieve even a single frame
          break

    # log some basic video stream after we've seen the first frame
    if nFrame == 0:
      frameHeight, frameWidth, frameChannels = currFrame.shape
      print("Video resolution: {} x {} @ {} fps".format(frameWidth, frameHeight, fps), flush=True)
    winx, winy, winWidth, winHeight = cv.getWindowImageRect(windowName) #(x, y, w, h)

    # halt main loop while we wait for the user to draw the 2 rectangles where motiondetection will happen
    if selectRectanglesRequested:
      selectRectanglesRequested = False
      dirtyConfig = selectRectangles(currFrame, winWidth, winHeight, windowName, config, dirtyConfig)

    # detect each meaningful motion in the input and output rectangles
    # group each of those motions (class MotionDetection) into photon-to-photon measurements (class Latency)
    inMotionDetected = False
    outMotionDetected = False
    frameRender = currFrame.copy()
    if prevFrame is not None:
      # run motion detectors
      inMotionDetected,   inScore,  inDiff = detectMotion(prevFrame, currFrame, nFrame, historySize,  config['inputRectangle'], frameWidth, frameHeight, config,  inScores, config[ 'inThreshold'])
      outMotionDetected, outScore, outDiff = detectMotion(prevFrame, currFrame, nFrame, historySize, config['outputRectangle'], frameWidth, frameHeight, config, outScores, config['outThreshold'])

      # render the frame-to-frame image difference in their input/output rectangles
      drawSubImage(frameRender,  inDiff, 10, config[ 'inputRectangle'], frameWidth, frameHeight)
      drawSubImage(frameRender, outDiff, 10, config['outputRectangle'], frameWidth, frameHeight)

      # start tracking a new Latency measurement when things have settled down (indicated by cooldown reaching zero), and save the previous Latency if it was valid
      cooldownPrev = cooldown
      if not pauseOnce: cooldown = cooldown - spf # advance cooldown
      cooldownFinished = cooldownPrev > 0 and cooldown < 0
      cooldown = max(0, cooldown)
      if cooldownFinished or latency.total(spf) > maxLatency*1.25:
        if latency.outDetectedFirst:
          # we have both input motion and output motion, so let's store this Latency measurement
          delaySecs = spf*(latency.outDetectedFirst - latency.inDetectedFirst)
          latencies.append(latency)
          while(len(latencies) > 30):
            latencies.pop(0)
        # start a new Latency measurement
        latency = Latency()
        inScorePeak = 0
        outScorePeak = 0

      # we detected motion in the Input rectangle
      if inMotionDetected:
        latency.inDetectedLast = nFrame
        if latency.inDetectedFirst: # we are already keeping track of a Latency instance
          # bump the cooldown
          cooldown = max(cooldown, 0.1)
        else: # we're not tracking a Latency instance, start gathering its data!
          latency.inDetectedFirst = nFrame
          latency.startTime = currTime
          inScorePeakValid = inScore # purely for UI purposes
          # bump the cooldown so we don't start measuring again too soon (mixing together two measurements if the user is impatiently using the input device)
          # if the latencies are high, cooldown needs to be high too. we first start at around 800ms of cooldown, and gradually shift towards the real latency average as we gather more and more data
          avgLatency = sum([maxLatency]+[i.secs(spf) for i in latencies]) / (1+len(latencies))
          cooldown = max(cooldown, max(0.3, avgLatency))
        motionDetection = MotionDetection("in", inScore, config['inThreshold'], spf*(latency.inDetectedLast-latency.inDetectedFirst))
        latency.motionDetections.append(motionDetection)

      # we detected motion in the Input rectangle
      if outMotionDetected:
        if latency.inDetectedFirst:
          latency.outDetectedLast = nFrame
          if latency.outDetectedFirst:
            # we've detected Output motion more than once after the original Input motion
            # let's wait a tiny bit for Output to settle down
            cooldown = max(cooldown, 0.1)
          else:
            # we've detected Output motion for the first time after detecting Input motion
            latency.outDetectedFirst = nFrame
            outScorePeakValid = outScore
            # let's wait a in case the Output motion is lengthy and bumps the cooldown even more
            cooldown = max(cooldown, 0.2)
          motionDetection = MotionDetection("out", outScore, config['outThreshold'], spf*(latency.outDetectedLast-latency.inDetectedFirst))
          latency.motionDetections.append(motionDetection)
        else: # we got motion on the Output rectangle before we got any Input motion
          cooldown = max(cooldown, 0.1) # the output imagery hasn't settled yet, or is unstable. wait a bit

      if latency.inDetectedFirst: # we are measuring latency atm
        latency.lastFrame = nFrame
      if not inMotionDetected: inScorePeak = max(inScore, inScorePeak)
      if not outMotionDetected: outScorePeak = max(outScore, outScorePeak)

    # draw In and OUT rects
    frameRender = cv.resize(frameRender, (winWidth, winHeight))
    frameRender = drawRect(frameRender, config[ 'inputRectangle'],  inputColor, 3 if  inMotionDetected else 1, winWidth, winHeight)
    frameRender = drawRect(frameRender, config['outputRectangle'], outputColor, 3 if outMotionDetected else 1, winWidth, winHeight)

    line = 0

    # draw IN and OUT bars
    if config['inputRectangle'] is not None and config['outputRectangle'] is not None:
      maxValue = 50
      x = 100
      w = max(0, min(300, winWidth - x - 20))
      drawText(frameRender, "Input motion", 5, line=line, color=inputColor, colorbg=white)
      frameRender = drawBar(frameRender, x, getLineY(line)-4, w, maxValue, config['inThreshold'], inScore, inScorePeak, inScorePeakValid, inputColor)
      line += 1
      drawText(frameRender, "Output motion", 5, line=line, color=outputColor, colorbg=white)
      frameRender = drawBar(frameRender, x, getLineY(line)-4, w, maxValue, config['outThreshold'], outScore, outScorePeak, outScorePeakValid, outputColor)
      line += 1

    line = drawHeaderText(frameRender, currTime, videoEnded, line)

    # draw stats text
    if advanced:
      drawText(frameRender, "{}x{}@{}Hz | (w)ebcam{} | (q)uality: {} | (c)onfig | (m)otion detector: {}{}".format(
        frameWidth, frameHeight, int(fps),
        " #{}".format(videopath) if type(videopath) is int else "",
        config['quality'],
        config['detectMode'],
        " | (+,-)edge threshold {:.1f}".format(config['edgeThreshold']) if config['detectMode'] != "abs" else ""
      ), 5, line=line)
      line += 1
      drawText(frameRender, "frame {:05} | (1,2)input {:.1f}/{:.1f}/{:.1f} | (3,4)output {:.1f}/{:.1f}/{:.1f}".format(
        nFrame,
        inScore, config['inThreshold'], inScorePeakValid,
        outScore, config['outThreshold'], outScorePeakValid,
      ), 5, line=line)
      line += 1
      drawText(frameRender, "Distributed under MIT license (see license.txt)", 1, y=winHeight-16, fontScale=0.3)
      drawText(frameRender, "Copyright 2021 Bruno Gonzalez Campo (@stenyak)", 1, y=winHeight-4, fontScale=0.3)

    # draw measured latencies
    yStart = getLineY(line) - 8
    n = 0
    for i in latencies: # history
      secs = i.secs(spf)
      total = i.total(spf)
      frameRender = drawLatency(frameRender, advanced, i.startTime, spf, n, yStart, secs, total, black, darkgrey, white, "{} ms".format(int(secs*1000)), i.motionDetections)
      n += 1
    if latency.inDetectedFirst>0 and not videoEnded: # current
      secs = latency.secs(spf)
      total = latency.total(spf)
      frameRender = drawLatency(frameRender, advanced, latency.startTime, spf, n, yStart, secs, total, black, darkgrey, white, "{} ms".format(int(secs*1000)), latency.motionDetections, cooldown)
      n += 1
    # average+ready
    rectsDefined = config['inputRectangle'] is not None and config['outputRectangle'] is not None
    label   = "(R)estart" if videoEnded else ((  "Wait" if cooldown > 0 else "Ready") if rectsDefined else "Please (s)elect motion rectangles")
    colorbg =       green if videoEnded else ((darkgrey if cooldown > 0 else   green) if rectsDefined else red)
    secs = ((sum([i.secs(spf) for i in latencies])/len(latencies) if latencies else 0.1) if rectsDefined else 0.25)
    total = secs
    frameRender = drawLatency(frameRender, advanced, currTime, spf, n, yStart, secs, total, black, colorbg, white, "{}{}".format("{} avg. ".format(int(secs*1000)) if latencies else "", label), [])
    n += 1

    # update window
    cv.imshow(windowName, frameRender)

    # wait and process input
    key = cv.waitKey(0 if paused else 1)
    pauseOnce = False
    breakRequested, retry, paused, pauseOnce, dirtyConfig, selectRectanglesRequested, webcamNextRequested  = processKeypress(key, config, retry, paused, pauseOnce, dirtyConfig, selectRectanglesRequested, cap)

    if dirtyConfig:
      saveConfig(videopath, config)

    # exit main loop (it then may or may not be restarted, based on user request)
    if breakRequested:
      break

  cap.release()
  cv.destroyAllWindows()
  return retry, webcamNextRequested

# ugh, the design/logic of main function calling should be cleaned up from those nasty vars everywhere, buuuuut i'm tired.
if __name__ == "__main__":
  webcamNextRequested = False
  retry = True
  while retry or webcamNextRequested: # user may want to restart the video from scratch (e.g. after reaching end of video, or switching webcam source)
    retry, webcamNextRequested = main(webcamNextRequested)
    print("")
