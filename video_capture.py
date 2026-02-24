import cv2

video_capture = cv2.VideoCapture(0)

frameWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate = int(video_capture.get(cv2.CAP_PROP_FPS))

# VideoWriter Objetct
fourccCode = cv2.VideoWriter_fourcc(*'mp4v')
videoFileName = 'sample_vid.mp4'
videoDimension = (frameWidth, frameHeight)

recordedVideo = cv2.VideoWriter(videoFileName, fourccCode, frameRate, videoDimension)

#640 by 480

while video_capture.isOpened():
    ret, frame = video_capture.read()
    cv2.imshow("video", frame)
    recordedVideo.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
recordedVideo.release()
cv2.destroyAllWindows()
