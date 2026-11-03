import cv2
from deepface import DeepFace

def get_cognitive_response(emotion):
    """Provides empathetic cognitive advice based on facial expression analysis."""
    responses = {
        'happy': "😊 Great to see you in a positive mood! Keep up the good work.",
        'sad': "💙 You seem a bit low. Take a short break or listen to soothing music.",
        'angry': "🧘 Take a deep breath. Pausing for a minute can help clear your mind.",
        'fear': "🛡️ Take a moment to relax. Focus on what you can control right now.",
        'surprise': "😲 Something unexpected happened? Stay calm and evaluate your options.",
        'neutral': "😐 Focused and steady. Perfect mindset for productive work.",
        'disgust': "🤔 Uncomfortable situation? Take a step back and reset."
    }
    return responses.get(emotion.lower(), "🤖 Ready to assist you.")

def run_emotion_assistant():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Emotion-Aware Cognitive Assistant... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Perform facial emotion analysis
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(analysis, list):
                analysis = analysis[0]

            dominant_emotion = analysis['dominant_emotion']
            box = analysis['region']

            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            # Draw bounding box around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 0), 2)

            # Retrieve cognitive assistant guidance message
            msg = get_cognitive_response(dominant_emotion)

            # Display emotion state & advice on video frame
            cv2.putText(frame, f"Detected Emotion: {dominant_emotion.upper()}", 
                        (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Assistant: {msg}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            pass

        cv2.imshow("Emotion-Aware Cognitive Assistant", frame)

        # Press 'q' to close video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_emotion_assistant()