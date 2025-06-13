import cv2

def detect_cars(video_path):
    try:
        car_cascade=cv2.CascadeClassifier('haarcascade_car.xml')
        if car_cascade.empty():
            print("Error: Could not load car cascade classifier.")
            print("Please make sure 'haarcascade_car.xml' is in the same directory as the script.")
            return
    except Exception as e:
        print(f"An error occurred while loading the cascade file:  {e}")
        return 
    
    # Open the video file
    try:
        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
    except Exception as e:
        print(f"An error occurred while opening the video file: {e}")
        return 
    print("Proccessing video... Press 'q' to quit.")

    # Loop through the video frames 
    while cap.isOpened():
        #Read a frame from the video
        ret, frame=cap.read()

        #If the frame was not read correctly (e.g., end of video), break the loop
        if not ret:
            print("End of video reached or cannot read frame.")
            break

        # Convert the frame to grayscale for the detection process
        # Haar Cascades work better on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the grayscale frame
        # The detectMultiScale function finds objects of different sizes.
        # - scaleFactor: How much the image size is reduced at each image scale.
        # - minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        #   Higher value results in fewer detections but with higher quality.
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        #Fraw a rectangle around each detected car 
        for(x, y, w, h) in cars:
            #Draw the rectangle on the original color frame
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            #Add a label "Car" above the rectangle
            cv2.putText(frame, 'Car', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #Display the resulting frame in a window 
        cv2.imshow('Car Detection', frame)

        #Wait for the 'q' key to be pressed to exit the loop 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    #Release the video capture object and destroy all OpenCv windows
    cap.release()
    cv2.destroyAllWindows()
    print("Vİdeo proccessing finished and windows closed.")

#--Main Execution-- 
if __name__ =="__main__":
    #Replace 'trafficjam.mov' with the path to your video file.
    # If you want to use a live webcam feed, change this to 0, like this: detect_cars(0)
    video_file='trafficjam.mov'
    detect_cars(video_file)  
