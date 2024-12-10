import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

def getAngle(a, b, c):
    angle  = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    if angle >180:
        angle = 360-angle
    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    st.title("Reps Counter")

    # Sidebar controls
    st.sidebar.title("Controls")
    start_camera = st.sidebar.button("Start Camera")
    stop_camera = st.sidebar.button("Stop Camera")

    col1, col2 = st.columns([3, 1])  # Camera feed in the wider column
    with col1:
        st.subheader("Camera Feed")
        camera_feed = st.empty()  # Placeholder for video

    with col2:
        st.subheader("Details")
        counter_placeholder = st.empty()


    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if start_camera:
        st.session_state.camera_running = True

    if stop_camera:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        # Video capture
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam

        if not cap.isOpened():
            st.error("Error: Could not open the webcam.")
            return

        st.text("Press 'Stop Camera' to stop the feed.")

        # Stream video
        frame_window = st.image([])  # Placeholder for video frames

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            counter = 0
            stage = None
            while st.session_state.camera_running:
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(frame)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Get  real-time landmarks
                    try:
                        landmarks = result.pose_landmarks.landmark

                        # Find the coordinates of shoulder , elbow and wrist
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        # find the angle between three points in 2D coordinate system
                        langle = getAngle(left_shoulder, left_elbow, left_wrist)
                        rangle = getAngle(right_shoulder,right_elbow,right_wrist)

                        # Put the value of angle in The actual position
                        cv2.putText(frame,
                                    str(langle),
                                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)

                        cv2.putText(frame,
                                    str(rangle),
                                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)

                        if langle >160 and rangle > 160:
                            stage = "down"
                        if langle <30 and rangle < 30 and stage == "down":
                            stage = "up"
                            counter += 1

                    except:
                        pass

                    cv2.rectangle(frame, (0, 0), (300, 74), (245, 117, 16), -1)

                    cv2.putText(frame, 'Reps', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame, 'Stage', (105, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, stage, (120, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                    # Draw connections
                    # mp_drawing.draw_landmarks(frame,
                    #                           result.pose_landmarks,
                    #                           mp_pose.POSE_CONNECTIONS,
                    #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                    #                                                  circle_radius=2),
                    #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                    #                                                  circle_radius=2))
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    counter_placeholder.markdown(f"Reps Count: {counter}")

                    camera_feed.image(frame, use_container_width=True)
                    # cv2.imshow("frame", frame)
                    # if cv2.waitKey(10) & 0xFF == ord('q'):
                    #     break


        # Release the video capture when done

        cap.release()
        cv2.destroyAllWindows()
    else:
        st.text("Click 'Start Camera' to begin capturing video.")


if __name__ == "__main__":
    main()
