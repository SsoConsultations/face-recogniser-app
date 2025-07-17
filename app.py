import streamlit as st
import face_recognition
import os
import numpy as np
import cv2
from PIL import Image # Import Pillow for image manipulation
import firebase_admin
from firebase_admin import credentials, storage
import json
import tempfile # For creating temporary files

try:
    FIREBASE_CREDENTIALS_JSON = st.secrets["FIREBASE_CONFIG"]
    FIREBASE_STORAGE_BUCKET = st.secrets["FIREBASE_STORAGE_BUCKET"]
except KeyError:
    st.warning("Firebase credentials not found in Streamlit Secrets. "
               "Attempting to load from 'serviceAccountKey.json' for local development. "
               "Ensure this file is in your .gitignore.")
    try:
        # Fallback for local development if secrets.toml is not used
        with open("serviceAccountKey.json") as f:
            FIREBASE_CREDENTIALS_JSON = f.read()
        # IMPORTANT: Manually set your storage bucket here if using local file
        FIREBASE_STORAGE_BUCKET = "face-recogniser-app.appspot.com" # <--- REPLACE WITH YOUR ACTUAL BUCKET IF USING LOCAL FILE
    except FileNotFoundError:
        st.error("Firebase 'serviceAccountKey.json' not found and Streamlit Secrets not configured. "
                 "Please set up Firebase credentials to proceed.")
        st.stop() # Stop the app if credentials are not available

# Initialize Firebase Admin SDK only once
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
        firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_STORAGE_BUCKET})
        st.success("Firebase initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")
        st.stop()

# --- Configuration for storage path within Firebase Storage ---
FIREBASE_KNOWN_FACES_PATH = 'known_faces/' # Path within your Firebase Storage bucket

# --- Data Storage (use st.cache_resource for efficiency) ---
# This function loads known faces from Firebase Storage.
# It's cached to avoid re-downloading and re-encoding on every Streamlit rerun,
# unless explicitly cleared (e.g., when a new face is added).
@st.cache_resource(show_spinner=False)
def load_known_faces_from_firebase(_=None): # Added _=None for cache invalidation trick
    st.info("Loading known faces from Firebase... This might take a moment.")
    known_face_encodings = []
    known_face_names = []

    try:
        bucket = storage.bucket()
        # List all blobs (files) within the specified path in Firebase Storage
        blobs = bucket.list_blobs(prefix=FIREBASE_KNOWN_FACES_PATH)

        found_faces = 0
        for blob in blobs:
            # Only process image files
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract person's name from the blob path.
                # Assumes structure like 'known_faces/person_name/image.jpg'
                parts = blob.name.split('/')
                if len(parts) >= 2:
                    person_name_raw = parts[-2]
                    person_name = person_name_raw.replace("_", " ").title() # Clean up name for display

                    # Create a temporary file to download the image
                    # Using tempfile.NamedTemporaryFile ensures unique filenames and handles cleanup
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(blob.name)[1]) as temp_file:
                        temp_image_path = temp_file.name
                        blob.download_to_filename(temp_image_path)

                    try:
                        image = face_recognition.load_image_file(temp_image_path)
                        face_locations = face_recognition.face_locations(image)
                        face_encodings = face_recognition.face_encodings(image, face_locations)

                        if face_encodings:
                            known_face_encodings.append(face_encodings[0])
                            known_face_names.append(person_name)
                            found_faces += 1
                        # else:
                        #     st.warning(f"No face found in {blob.name}. Skipping.")
                    except Exception as img_proc_e:
                        st.error(f"Error processing image {blob.name} from Firebase: {img_proc_e}")
                    finally:
                        # Ensure temporary file is deleted
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
        
        if found_faces == 0:
            st.warning("No known faces found in Firebase Storage. Please add faces via the Admin Panel.")
        else:
            st.success(f"Finished loading known faces from Firebase. Total known faces: {found_faces}")

    except Exception as e:
        st.error(f"Error loading known faces from Firebase: {e}")
        st.warning("Please ensure your Firebase Storage bucket is correctly configured and has read permissions.")
        # If there's an error, return empty lists to prevent app from crashing
        return [], []
        
    return known_face_encodings, known_face_names

# Initialize global variables at module level.
# These will be populated by the cached function call.
known_face_encodings = []
known_face_names = []

# Load faces once when the app starts or is re-run due to cache invalidation.
# The initial call to load_known_faces_from_firebase will populate the global lists.
known_face_encodings, known_face_names = load_known_faces_from_firebase()


# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    """
    Detects faces in an RGB image, recognizes them against known faces,
    and draws bounding boxes and labels.
    Args:
        frame_rgb (numpy.ndarray): The input image in RGB format.
        known_encodings (list): List of known face encodings.
        known_names (list): List of names corresponding to known face encodings.
    Returns:
        numpy.ndarray: The image with detected faces, boxes, and labels drawn, in BGR format.
    """
    frame_rgb = np.copy(frame_rgb)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    # Convert the image to BGR for OpenCV drawing functions
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr # Return original frame if no faces found

    # Iterate through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown" # Default name if no match is found

        if known_encodings: # Only compare if there are known faces in the database
            # Compare current face encoding with all known face encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            # Calculate the distance to each known face (lower distance means better match)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the best match (smallest distance)
            best_match_index = np.argmin(face_distances)
            
            # If the best match is actually a match (within a certain tolerance)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                # Optional: If no exact match, consider the closest match if within a threshold
                # 0.6 is a common threshold for face_recognition library. Adjust as needed.
                if face_distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]

        # --- Drawing rectangles and labels on the image ---
        box_padding = 15
        base_label_height = 25
        text_y_offset = 10
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1

        # Extend the bounding box slightly for better visual appeal
        top_ext = max(0, top - box_padding)
        right_ext = min(frame_bgr.shape[1], right + box_padding)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding)
        left_ext = max(0, left - box_padding)

        # Draw the face bounding box (green rectangle)
        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2)

        # Calculate text size for dynamic label width
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        label_width = text_width + (box_padding * 2)
        label_height = max(base_label_height, text_height + (text_y_offset * 2))

        # Position the label rectangle below the face box
        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width) # Ensure label is at least as wide as the box

        # Adjust label position if it goes out of image bounds
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0: # If still goes above, set to 0
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width) # Ensure label is not out of bounds left

        # Draw the label background rectangle (filled green)
        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        # Calculate text position to center it within the label rectangle
        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        # Put the recognized name text on the image (black text)
        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home' # 'home', 'user_login', 'admin_login'

# --- Home Page ---
if st.session_state.page == 'home':
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        try:
            # Ensure 'sso_logo.jpg' is in the same directory as your app.py
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
        st.error("No known faces loaded. Please ensure faces are added to Firebase Storage via the Admin Panel.")

    st.sidebar.header("Choose Input Method")
    option = st.sidebar.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"), key="user_input_option")

    if option == "Live Webcam Recognition":
        st.subheader("Live Webcam Face Recognition")
        st.info("Allow camera access. Take a picture, and the app will detect/recognize faces.")

        camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="user_camera_input")

        if camera_image is not None:
            with st.spinner("Processing live image..."):
                # Convert uploaded file bytes to a numpy array (BGR format for OpenCV)
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for face_recognition

                # Process the image to detect and recognize faces
                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for Streamlit display

            st.image(processed_img_rgb, caption="Processed Live Image", use_column_width=True)
            st.success("Face detection and recognition complete!")
        else:
            st.warning("Waiting for webcam input. Click 'Take Photo' above.")

    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                # Convert uploaded file bytes to a numpy array (BGR format for OpenCV)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for face_recognition

                st.image(img_rgb, caption="Original Uploaded Image", use_column_width=True)

                # Process the image to detect and recognize faces
                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for Streamlit display

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

    # IMPORTANT: Replace "admin123" with a more secure authentication method for production!
    # For example, integrate with Firebase Authentication or a proper user management system.
    if admin_password == "admin123":
        st.success("Welcome, Admin!")

        st.subheader("Add New Faces to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_image = st.file_uploader("Upload Image of New Face:", type=["jpg", "jpeg", "png"], key="new_face_image_uploader")

        if st.button("Add Face to Database", key="add_face_btn"):
            if new_face_name and new_face_image:
                try:
                    # Clean up name for storage path (e.g., "John Doe" -> "john_doe")
                    person_dir_name = new_face_name.replace(" ", "_").lower()
                    
                    # Read the uploaded image file as bytes
                    image_bytes = new_face_image.read()
                    
                    # Convert bytes to numpy array (BGR) for OpenCV and then to RGB for face_recognition
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_to_encode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    image_to_encode_rgb = cv2.cvtColor(image_to_encode, cv2.COLOR_BGR2RGB)

                    # Check if a face can be detected in the uploaded image
                    face_locations = face_recognition.face_locations(image_to_encode_rgb)

                    if face_locations:
                        bucket = storage.bucket() # Get Firebase Storage bucket reference
                        
                        # Generate a unique filename for the image within the person's folder
                        # List existing blobs in the person's folder to count for unique naming
                        existing_blobs = bucket.list_blobs(prefix=f"{FIREBASE_KNOWN_FACES_PATH}{person_dir_name}/")
                        num_existing_files = sum(1 for _ in existing_blobs if _.name.lower().endswith(('.png', '.jpg', '.jpeg')))
                        
                        # Construct the full path for the new image in Firebase Storage
                        image_filename = f"{person_dir_name}_{num_existing_files + 1}.jpg" # Example: john_doe_1.jpg
                        firebase_path = f"{FIREBASE_KNOWN_FACES_PATH}{person_dir_name}/{image_filename}"

                        blob = bucket.blob(firebase_path) # Create a blob (file) reference
                        
                        # Upload the image directly from bytes
                        blob.upload_from_string(image_bytes, content_type='image/jpeg') # Specify content type

                        st.info(f"Analyzing {new_face_name}'s image for encoding and updating database...")

                        # Clear the cache for load_known_faces_from_firebase to force a reload
                        load_known_faces_from_firebase.clear()

                        # Re-load known faces from Firebase; this will update the global lists
                        # The '_=np.random.rand()' is a trick to force cache invalidation for st.cache_resource
                        # Removed 'global' keyword here as it's not needed for re-assignment of global variables
                        known_face_encodings, known_face_names = load_known_faces_from_firebase(_=np.random.rand())

                        st.success(f"Successfully added '{new_face_name}' to the known faces database on Firebase! ✅")
                        st.rerun() # Rerun the app to refresh the UI and known faces list
                    else:
                        st.error(f"No face found in the uploaded image for '{new_face_name}'. Please upload an image with a clear face.")

                except Exception as e:
                    st.error(f"Error adding face to Firebase: {e}")
                    st.exception(e) # Display full traceback for debugging

            else:
                st.warning("Please provide both a name and upload an image.")

        st.subheader("Current Known Faces 📋")
        if known_face_names:
            # Display current known faces with unique names, sorted alphabetically
            for name in sorted(set(known_face_names)):
                st.write(f"- *{name}*")
        else:
            st.info("No faces currently registered in the database.")

    else:
        if admin_password: # Only show error if user actually typed something
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤ using face_recognition, OpenCV, and Streamlit.")
