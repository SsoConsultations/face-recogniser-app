import os
import json
import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image  # Import Pillow for image manipulation
import firebase_admin
from firebase_admin import credentials, firestore, storage

# --- Firebase Initialization ---
# Load Firebase credentials from environment variable
firebase_service_account = os.getenv('FIREBASE_SERVICE_ACCOUNT')
cred = credentials.Certificate(json.loads(firebase_service_account))
firebase_admin.initialize_app(cred, {
    'storageBucket': 'face-recogniser-app.appspot.com'  # Replace with your Firebase Storage bucket name
})

# Initialize Firestore
db = firestore.client()
bucket = storage.bucket()

# --- Data Storage (use st.cache_resource for efficiency) ---
@st.cache_resource
def load_known_faces_from_firestore():
    st.info("Loading known faces from Firestore... This might take a moment.")
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    # Fetch known faces from Firestore
    faces_ref = db.collection('known_faces').stream()
    for face in faces_ref:
        data = face.to_dict()
        known_face_names.append(data['name'])
        # Load the image from Firebase Storage
        image_blob = bucket.blob(data['image_path'])
        image_data = image_blob.download_as_bytes()
        image = face_recognition.load_image_file(image_data)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])

    st.success(f"Finished loading known faces from Firestore. Total known faces: {len(known_face_encodings)}")
    return known_face_encodings, known_face_names

# Initialize global variables at module level
known_face_encodings = []
known_face_names = []

# Load faces once when the app starts
known_face_encodings, known_face_names = load_known_faces_from_firestore()

# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    frame_rgb = np.copy(frame_rgb)

    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr  # Return original frame if no faces found

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if known_encodings:  # Only compare if there are known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                if face_distances[best_match_index] < 0.6:  # Adjust threshold as needed
                    name = known_names[best_match_index]

        # Drawing rectangles and labels
        box_padding = 15
        base_label_height = 25
        text_y_offset = 10
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1

        top_ext = max(0, top - box_padding)
        right_ext = min(frame_bgr.shape[1], right + box_padding)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding)
        left_ext = max(0, left - box_padding)

        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2)

        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        label_width = text_width + (box_padding * 2)
        label_height = max(base_label_height, text_height + (text_y_offset * 2))

        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width)

        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0:
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width)

        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # 'home', 'user_login', 'admin_login'

# --- Home Page ---
if st.session_state.page == 'home':
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        try:
            st.image("sso_logo.jpg", width=300)
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found. Please ensure it's in the same directory.")
            st.markdown("## SSO Consultants")

    st.markdown("<h2 style='text-align: center;'>SSO Consultants Face Recogniser 🕵‍♂</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Please choose your login type.</h3>", unsafe_allow_html=True)

    col1_btn, col2_btn, col3_btn, col4_btn = st.columns([1, 0.7, 0.7, 1])

    with col2_btn:
        if st.button("Login as User", key="user_login_btn", help="Proceed to face recognition for users"):
            st.session_state.page = 'user_login'
            st.rerun()

    with col3_btn:
        if st.button("Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities"):
            st.session_state.page = 'admin_login'
            st.rerun()

# --- User Login (Face Recognition) Page ---
elif st.session_state.page == 'user_login':
    st.title("Face Recognition App with Dynamic Labels 🕵‍♂")
    st.markdown("""
    This application performs face recognition from your live webcam or an uploaded image.
    The name labels will dynamically adjust their size to fit the recognized name!
    """)

    if not known_face_encodings:
        st.error("No known faces loaded. Please ensure your Firestore contains images with faces for training.")

    st.sidebar.header("Choose Input Method")
    option = st.sidebar.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"), key="user_input_option")

    if option == "Live Webcam Recognition":
        st.subheader("Live Webcam Face Recognition")
        st.info("Allow camera access. Take a picture, and the app will detect/recognize faces.")

        camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="user_camera_input")

        if camera_image is not None:
            with st.spinner("Processing live image..."):
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

            st.image(processed_img_rgb, caption="Processed Live Image", use_column_width=True)
            st.success("Face detection and recognition complete!")
        else:
            st.warning("Waiting for webcam input. Click 'Take Photo' above.")

    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption="Original Uploaded Image", use_column_width=True)

                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

            st.image(processed_img_rgb, caption="Processed Image with Faces", use_column_width=True)
            st.success("Face detection and recognition complete!")
        else:
            st.info("Please upload an image file using the browser button.")

    if st.button("⬅ Back to Home", key="user_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Login Page ---
elif st.session_state.page == 'admin_login':
    st.title("Admin Panel 🔒")
    st.markdown("This section is for *administrators* only.")

    admin_password = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")

    if admin_password == "admin123":  # *IMPORTANT: Replace with a more secure authentication method for production!*
        st.success("Welcome, Admin!")

        st.subheader("Add New Faces to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_image = st.file_uploader("Upload Image of New Face:", type=["jpg", "jpeg", "png"], key="new_face_image_uploader")

        if st.button("Add Face to Database", key="add_face_btn"):
            if new_face_name and new_face_image:
                # Save the image to Firebase Storage
                image_filename = f"{new_face_name.replace(' ', '_').lower()}.jpg"
                blob = bucket.blob(f'known_faces/{image_filename}')
                blob.upload_from_file(new_face_image)

                # Save the image path and name to Firestore
                db.collection('known_faces').add({
                    'name': new_face_name,
                    'image_path': f'known_faces/{image_filename}'  # Store the path in Firestore
                })

                load_known_faces_from_firestore.clear()
                known_face_encodings, known_face_names = load_known_faces_from_firestore()

                st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                st.rerun()
            else:
                st.warning("Please provide both a name and upload an image.")

        st.subheader("Current Known Faces 📋")
        if known_face_names:
            for name in sorted(set(known_face_names)):
                st.write(f"- *{name}*")
        else:
            st.info("No faces currently registered in the database.")

    else:
        if admin_password:  # Only show error if user actually typed something
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤ using face_recognition, OpenCV, and Streamlit.")
