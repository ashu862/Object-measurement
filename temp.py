import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Canny Edge Detection to find edges in the image
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are contours detected
    if contours:
        # Sort the contours by area (in descending order)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get the largest contour (the first one in the sorted list)
        contour = contours[0]

        # Get the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding rectangle on the original image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the width and height of the object
        cv2.putText(frame, f'W:{w}, H:{h}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Real Time Object Measurement', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
