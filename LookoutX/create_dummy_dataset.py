import os
import cv2

wifi_ip = '10.203.190.99'
port = '4747'

livestream_url = f'http://{wifi_ip}:{port}/video'
data_path = 'dataset'

frames = []
video = cv2.VideoCapture(livestream_url)

while True:
    # Read a frame from the video stream
    check, frame = video.read()

    # Waiting for 1ms
    key = cv2.waitKey(1)
    cv2.imshow('Frame', frame)

    # Check for the 'q' key to quit the program
    if key == ord('q'):
        img_idx = [int(x.split(".")[0]) for x in os.listdir(data_path)]
        last_img = sorted(img_idx)[-1]
        fname = os.path.join(data_path, f'{last_img + 1}.jpg')
        cv2.imwrite(fname, frame)
        print(f'Saved to {fname}')
        break

video.release()

cv2.destroyAllWindows()
