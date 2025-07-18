import os
import json
import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image # Import Pillow for image manipulation

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage

    # Load Firebase credentials from Streamlit secrets
    # The JSON string from secrets needs to be parsed into a dictionary
    firebase_service_account_json_str = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
    firebase_service_account_dict = json.loads(firebase_service_account_json_str)

    cred = credentials.Certificate(firebase_service_account_dict)

    # Check if Firebase app is already initialized to prevent re-initialization errors
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'storageBucket': st.secrets["FIREBASE_STORAGE_BUCKET"]
        })

    # Initialize Firestore and Storage bucket
    db = firestore.client()
    bucket = storage.bucket()

    firebase_initialized = True
except Exception as e:
    st.error(f"Failed to initialize Firebase. Please check your `FIREBASE_SERVICE_ACCOUNT` and `FIREBASE_STORAGE_BUCKET` secrets. Error: {e}")
    firebase_initialized = False

# --- Data Storage (use st.cache_resource for efficiency) ---
@st.cache_resource
def load_known_faces_from_firestore():
    """
    Loads known face encodings and names from Firestore and Firebase Storage.
    Uses st.cache_resource to cache the results for performance.
    """
    if not firebase_initialized:
        st.warning("Firebase not initialized. Cannot load known faces.")
        return [], []

    st.info("Loading known faces from Firestore... This might take a moment.")
    
    local_known_face_encodings = []
    local_known_face_names = []

    try:
        # Fetch known faces from Firestore
        faces_ref = db.collection('known_faces').stream()
        for face in faces_ref:
            data = face.to_dict()
            name = data.get('name')
            image_path = data.get('image_path')

            if name and image_path:
                local_known_face_names.append(name)
                # Load the image from Firebase Storage
                try:
                    image_blob = bucket.blob(image_path)
                    image_data = image_blob.download_as_bytes()
                    # face_recognition.load_image_file can directly handle bytes
                    image = face_recognition.load_image_file(image_data)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        local_known_face_encodings.append(face_encodings[0])
                    else:
                        st.warning(f"No face detected in image for '{name}' at '{image_path}'. Skipping.")
                except Exception as e:
                    st.error(f"Error loading image for '{name}' from Storage at '{image_path}': {e}")
            else:
                st.warning(f"Skipping malformed known face entry: {data}")

        st.success(f"Finished loading known faces from Firestore. Total known faces: {len(local_known_face_encodings)}")
    except Exception as e:
        st.error(f"Error fetching known faces from Firestore: {e}")
        st.warning("Ensure your Firestore 'known_faces' collection exists and contains valid data.")
    
    return local_known_face_encodings, local_known_face_names

# Initialize global variables at module level
known_face_encodings = []
known_face_names = []

# Load faces once when the app starts, only if Firebase is initialized
if firebase_initialized:
    known_face_encodings, known_face_names = load_known_faces_from_firestore()

# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    """
    Detects faces in an RGB image, recognizes them against known encodings,
    and draws bounding boxes and labels on the image.
    """
    frame_rgb_copy = np.copy(frame_rgb)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame_rgb_copy)
    face_encodings = face_recognition.face_encodings(frame_rgb_copy, face_locations)

    # Convert the image from RGB (face_recognition) to BGR (OpenCV) for drawing
    frame_bgr = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr  # Return original frame if no faces found

    # Loop through each face found in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        # Only compare if there are known faces loaded
        if known_encodings:
            # Compare current face with known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            
            # If a match is found and the distance is below a threshold, set the name
            # The threshold (0.6) can be adjusted: lower for stricter match, higher for looser match
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
            # Optional: If no exact match but close enough, still assign the closest known name
            # This can help with variations, but might increase false positives if threshold is too high
            elif not matches[best_match_index] and face_distances[best_match_index] < 0.5: # A slightly stricter threshold for 'closest unknown'
                 name = f"Likely {known_names[best_match_index]}"


        # --- Drawing rectangles and labels with dynamic sizing ---
        box_padding = 15 # Padding around the face box
        base_label_height = 25 # Minimum height for the name label
        text_y_offset = 10 # Vertical offset for text within the label
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1

        # Extend the bounding box for better visual appeal
        top_ext = max(0, top - box_padding)
        right_ext = min(frame_bgr.shape[1], right + box_padding)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding)
        left_ext = max(0, left - box_padding)

        # Draw a box around the face
        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2) # Green box

        # Calculate text size to dynamically size the label background
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        
        # Calculate label dimensions
        label_width = text_width + (box_padding * 2)
        label_height = max(base_label_height, text_height + (text_y_offset * 2))

        # Position the label below the face box
        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width) # Ensure label is at least as wide as face box

        # Adjust label position if it goes out of frame boundaries
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0: # If image is too small, just put it at top
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width) # Shift left if it goes off right edge

        # Draw a filled rectangle for the name label background
        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED) # Green background

        # Put the name text on the label
        text_x = label_left + (label_right - label_left - text_width) // 2 # Center text horizontally
        text_y = label_top + (label_height + text_height) // 2 - baseline # Center text vertically
        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness) # Black text

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
        # Placeholder for logo. Ensure 'sso_logo.jpg' is in your app's directory.
        try:
            st.image("https://placehold.co/300x100/000000/FFFFFF?text=SSO+Logo", width=300) # Using a placeholder for now
            # st.image("sso_logo.jpg", width=300) # Uncomment this if you have the actual image
        except Exception: # Catch any error, including FileNotFoundError if local image is used
            st.warning("Logo image 'sso_logo.jpg' not found. Using a placeholder.")
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

    if not firebase_initialized:
        st.error("Cannot proceed. Firebase failed to initialize.")
    elif not known_face_encodings:
        st.info("No known faces loaded. All detected faces will be labeled 'Unknown'.")

    st.sidebar.header("Choose Input Method")
    option = st.sidebar.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"), key="user_input_option")

    if option == "Live Webcam Recognition":
        st.subheader("Live Webcam Face Recognition")
        st.info("Allow camera access. Take a picture, and the app will detect/recognize faces.")

        camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="user_camera_input")

        if camera_image is not None:
            with st.spinner("Processing live image..."):
                try:
                    # Read bytes from camera input
                    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                    # Decode image from bytes to OpenCV BGR format
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    # Convert BGR to RGB for face_recognition library
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # Process the image for faces
                    processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                    # Convert back to RGB for Streamlit display
                    processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                    st.image(processed_img_rgb, caption="Processed Live Image", use_column_width=True)
                    st.success("Face detection and recognition complete!")
                except Exception as e:
                    st.error(f"Error processing camera image: {e}")
        else:
            st.warning("Waiting for webcam input. Click 'Take Photo' above.")

    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                try:
                    # Read bytes from uploaded file
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    # Decode image from bytes to OpenCV BGR format
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    # Convert BGR to RGB for face_recognition library
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    st.image(img_rgb, caption="Original Uploaded Image", use_column_width=True)

                    # Process the image for faces
                    processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                    # Convert back to RGB for Streamlit display
                    processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                    st.image(processed_img_rgb, caption="Processed Image with Faces", use_column_width=True)
                    st.success("Face detection and recognition complete!")
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
        else:
            st.info("Please upload an image file using the browser button.")

    if st.button("⬅ Back to Home", key="user_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Login Page ---
elif st.session_state.page == 'admin_login':
    st.title("Admin Panel 🔒")
    st.markdown("This section is for *administrators* only.")

    if not firebase_initialized:
        st.error("Cannot proceed. Firebase failed to initialize.")
        if st.button("⬅ Back to Home", key="admin_back_btn_no_firebase"):
            st.session_state.page = 'home'
            st.rerun()
        st.stop() # Stop execution if Firebase is not ready

    admin_password_input = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")

    # Get admin password from Streamlit secrets
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "default_admin_password_if_not_set") # Provide a fallback

    if admin_password_input == ADMIN_PASSWORD:
        st.success("Welcome, Admin!")

        st.subheader("Add New Faces to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_image = st.file_uploader("Upload Image of New Face:", type=["jpg", "jpeg", "png"], key="new_face_image_uploader")

        if st.button("Add Face to Database", key="add_face_btn"):
            if new_face_name and new_face_image:
                with st.spinner("Checking image for faces and uploading..."):
                    try:
                        # Read image bytes for face detection check
                        file_bytes = np.asarray(bytearray(new_face_image.read()), dtype=np.uint8)
                        # Decode and convert to RGB for face_recognition
                        img_rgb_for_check = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                        
                        # Check if a face is present in the uploaded image
                        face_encodings_new = face_recognition.face_encodings(img_rgb_for_check)
                        
                        if not face_encodings_new:
                            st.warning("No face detected in the uploaded image. Please upload an image with a clear face.")
                        else:
                            # If face detected, proceed with upload to Firebase Storage
                            # IMPORTANT: Reset file pointer to the beginning before uploading
                            new_face_image.seek(0) 
                            image_filename = f"{new_face_name.replace(' ', '_').lower()}_{os.urandom(4).hex()}.jpg" # Add random suffix to avoid name collisions
                            blob = bucket.blob(f'known_faces/{image_filename}')
                            blob.upload_from_file(new_face_image)

                            # Save the image path and name to Firestore
                            db.collection('known_faces').add({
                                'name': new_face_name,
                                'image_path': f'known_faces/{image_filename}'  # Store the path in Firestore
                            })

                            # Clear the cache for known faces and reload them
                            load_known_faces_from_firestore.clear()
                            known_face_encodings, known_face_names = load_known_faces_from_firestore()

                            st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                            st.rerun() # Rerun to clear form fields and update known faces list
                    except Exception as e:
                        st.error(f"Error adding face to database: {e}")
            else:
                st.warning("Please provide both a name and upload an image.")

        st.subheader("Current Known Faces 📋")
        if known_face_names:
            # Display unique names, sorted
            for name in sorted(list(set(known_face_names))):
                st.write(f"- *{name}*")
        else:
            st.info("No faces currently registered in the database.")

    else:
        if admin_password_input:  # Only show error if user actually typed something
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤ using face_recognition, OpenCV, and Streamlit.")
