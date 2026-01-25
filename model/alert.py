import cv2
from ultralytics import solutions
cap = cv2.VideoCapture(r"C:\Users\fyq\Downloads\抖音2025516-440265.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

from_email = "fanyueqing8@gmail.com"  # the sender email address
password = "umdj fmkz flkq hgvd"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
to_email = "yueqingfan@outlook.com"  # the receiver email address

# Initialize security alarm object
securityalarm = solutions.SecurityAlarm(
    show=True,  # display the output
    model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
    records=1,  # total detections count to send an email
)

securityalarm.authenticate(from_email, password, to_email)  # authenticate the email server

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = securityalarm(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows