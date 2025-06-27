# # ======================================================================================
# # Save Predictions in predictions.csv and Print Summary
# # ======================================================================================

# import os
# import cv2
# import numpy as np
# import csv
# from collections import defaultdict
# from tensorflow.keras.models import load_model

# # Paths
# test_dir = 'dataset/test'
# model_path = 'model/emotion_model.keras'
# output_csv = 'predictions.csv'

# # Emotions
# emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# label_to_emotion = {i: emotion for i, emotion in enumerate(emotions)}

# # Load model
# model = load_model(model_path)

# # Prepare CSV
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Filename', 'Actual', 'Predicted'])

#     # Track accuracy per emotion
#     total_counts = defaultdict(int)
#     correct_counts = defaultdict(int)

#     for emotion in emotions:
#         folder_path = os.path.join(test_dir, emotion)
#         if not os.path.exists(folder_path):
#             continue

#         for filename in os.listdir(folder_path):
#             if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 continue

#             filepath = os.path.join(folder_path, filename)
#             img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

#             if img is None:
#                 continue

#             img = cv2.resize(img, (48, 48)) / 255.0
#             img = img.reshape(1, 48, 48, 1)

#             prediction = model.predict(img, verbose=0)
#             predicted_label = np.argmax(prediction)
#             predicted_emotion = label_to_emotion[predicted_label]

#             # Write to CSV
#             writer.writerow([filename, emotion, predicted_emotion])

#             # Count stats
#             total_counts[emotion] += 1
#             if predicted_emotion == emotion:
#                 correct_counts[emotion] += 1

# # Print per-class summary
# print("\n📊 Per-Emotion Accuracy Summary:")
# for emotion in emotions:
#     total = total_counts[emotion]
#     correct = correct_counts[emotion]
#     acc = (correct / total) * 100 if total > 0 else 0
#     print(f"{emotion.capitalize():<10}: {correct}/{total} correct ({acc:.2f}%)")

# print(f"\n✅ Predictions saved in '{output_csv}'")


# # ======================================================================================
# # Shows the image and prediction in a window - Prints the actual vs predicted emotion
# # ======================================================================================

# # import os
# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.utils import to_categorical
# # import matplotlib.pyplot as plt

# # # Set path to test dataset
# # test_dir = 'dataset/test'

# # # List of emotions
# # emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# # emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}
# # label_to_emotion = {idx: emotion for emotion, idx in emotion_to_label.items()}

# # # Load model
# # model = load_model('model/emotion_model.keras')

# # # Loop through test images and predict
# # for emotion in emotions:
# #     folder = os.path.join(test_dir, emotion)
# #     print(f"\n📁 Predicting images in: {emotion.upper()}")
    
# #     for i, file in enumerate(os.listdir(folder)):
# #         if i >= 5:  # Show only 5 images per emotion to avoid overload
# #             break
# #         img_path = os.path.join(folder, file)
# #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# #         if img is None:
# #             continue
# #         img_resized = cv2.resize(img, (48, 48)) / 255.0
# #         input_img = img_resized.reshape(1, 48, 48, 1)

# #         pred = model.predict(input_img)
# #         pred_label = np.argmax(pred)
# #         pred_emotion = label_to_emotion[pred_label]

# #         # 🖼️ Show the image and prediction
# #         plt.imshow(img, cmap='gray')
# #         plt.title(f"Actual: {emotion} | Predicted: {pred_emotion}")
# #         plt.axis('off')
# #         plt.show()




# # ======================================================================================
# # LIVE WEB CAM EMOTION DETECTION
# # ======================================================================================

# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Load trained model (.keras format)
# # model = load_model("model/emotion_model.keras")
# model = load_model("C:/Users/Ali/Desktop/cvproc/Group01_AI_Project_BSCSF22/model/emotion_model.keras")


# # Labels from FER-2013
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Load OpenCV Haar cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         roi_gray = grayscale_frame[y:y+h, x:x+w]
#         roi_resized = cv2.resize(roi_gray, (48, 48))
#         roi_normalized = roi_resized / 255.0
#         roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
#         prediction = model.predict(roi_reshaped, verbose=0)
#         label = emotion_labels[np.argmax(prediction)]

#         # Draw rectangle and label
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("Facial Emotion Detection", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
