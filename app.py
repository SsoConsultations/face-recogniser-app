import streamlit as st
import face_recognition
import os
import numpy as np
import cv2
from PIL import Image
import json # For serializing/deserializing NumPy arrays to JSON strings

# --- Firebase Admin SDK Imports ---
import firebase_admin
from firebase_admin import credentials, firestore, auth # Import auth if you plan to use it later for user management

# --- Configuration ---
# Path to your Firebase service account key JSON file
SERVICE_ACCOUNT_KEY_PATH = 'serviceAccountKey.json'

# Firestore Collection Path
# This follows the recommended structure for public data in Canvas environment
# Replace 'your_app_id_here' with an actual unique ID for your app if not running in Canvas
# For local testing, you can use a fixed ID like 'my-face-recognition-app'
# In a true Canvas environment, __app_id would be injected, but for standalone, we define it.
APP_ID_FOR_FIRESTORE = os.environ.get("CANVAS_APP_ID", "sso-face-recogniser-app") # Use env var or a default

FIRESTORE_COLLECTION_PATH = f'artifacts/{APP_ID_FOR_FIRESTORE}/public/data/known_faces'

# --- Firebase Initialization (Happens once per Streamlit app run) ---
# Check if Firebase app is already initialized to prevent re-initialization errors
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
        st.success("Firebase Admin SDK initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize Firebase Admin SDK: {e}")
        st.warning("Please ensure 'serviceAccountKey.json' is in the correct directory and is valid.")
        st.stop() # Stop the app if Firebase cannot be initialized

db = firestore.client() # Get the Firestore client

# --- Data Storage (using Streamlit's cache and actual Firestore) ---
@st.cache_resource(ttl=300) # Cache for 5 minutes, can be cleared manually
def load_known_faces_from_firestore(_=None):
    """
    Loads known face encodings and names from Firebase Firestore.
    """
    st.info("Loading known faces from cloud database... This might take a moment.")
    
    known_face_encodings_local = []
    known_face_names_local = []

    try:
        # Query the Firestore collection
        docs = db.collection(FIRESTORE_COLLECTION_PATH).stream()

        for doc in docs:
            data = doc.to_dict()
            name = data.get('name')
            encodings_json = data.get('encodings', [])
            
            if name and encodings_json:
                for enc_json in encodings_json:
                    try:
                        # Convert JSON string back to NumPy array
                        encoding_list = json.loads(enc_json)
                        known_face_encodings_local.append(np.array(encoding_list))
                        known_face_names_local.append(name)
                    except json.JSONDecodeError:
                        st.warning(f"Could not decode encoding for {name} in document {doc.id}. Skipping.")
            else:
                st.warning(f"Skipping incomplete document {doc.id} in Firestore.")

    except Exception as e:
        st.error(f"Error loading known faces from database: {e}")
        st.warning("Please ensure your Firestore security rules are correctly set up.")

    st.success(f"Finished loading known faces. Total known faces: {len(known_face_encodings_local)}")
    return known_face_encodings_local, known_face_names_local

# Initialize global variables at module level
# These will be populated by the cached function
known_face_encodings = []
known_face_names = []

# Load faces once when the app starts or is re-run due to cache invalidation
known_face_encodings, known_face_names = load_known_faces_from_firestore()

# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    frame_rgb = np.copy(frame_rgb)
    
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr # Return original frame if no faces found

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if known_encodings: # Only compare if there are known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                # Optional: If no exact match, consider the closest match if within a threshold
                if len(face_distances) > 0 and face_distances[best_match_index] < 0.6: # Adjust threshold as needed
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
        label_right = max(right_ext, left_ext + label_width) # Ensure label is at least as wide as box

        # Adjust label position if it goes out of bounds
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0: # If still goes above, set to 0
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width) # Ensure label is not out of bounds left

        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home' # 'home', 'user_login', 'admin_login'

# Display User ID (MANDATORY for multi-user apps)
# For firebase-admin, the app itself is authenticated. We use a placeholder user ID.
st.sidebar.markdown(f"**Current User ID:** `Admin (via Service Account)`")
st.sidebar.markdown(f"**App ID:** `{APP_ID_FOR_FIRESTORE}`")


# --- Home Page ---
if st.session_state.page == 'home':
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        try:
            st.image("image_f1d98f.png", width=300) 
        except FileNotFoundError:
            st.warning("Logo image 'image_f1d98f.png' not found. Please ensure it's in the same directory.")
            st.markdown("## SSO Consultants")

    st.markdown("<h2 style='text-align: center;'>SSO Consultants Face Recogniser 🕵️‍♂️</h2>", unsafe_allow_html=True)
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
    st.title("Face Recognition App with Dynamic Labels 🕵️‍♂️")
    st.markdown("""
    This application performs face recognition from your live webcam or an uploaded image.
    The name labels will dynamically adjust their size to fit the recognized name!
    """)

    if not known_face_encodings:
        st.error("No known faces loaded from the database. Please ensure faces are added via the Admin Panel.")

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

# --- Admin Login Page (Placeholder) ---
elif st.session_state.page == 'admin_login':
    st.title("Admin Panel 🔒")
    st.markdown("This section is for **administrators** only.")

    admin_password = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")

    if admin_password == "admin123": # **IMPORTANT: Replace with a more secure authentication method for production!**
        st.success("Welcome, Admin!")

        st.subheader("Add New Faces to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_image = st.file_uploader("Upload Image of New Face:", type=["jpg", "jpeg", "png"], key="new_face_image_uploader")

        if st.button("Add Face to Database", key="add_face_btn"):
            if new_face_name and new_face_image:
                st.info(f"Analyzing {new_face_name}'s image and adding to database...")
                
                try:
                    # Convert uploaded file to a format face_recognition can use
                    file_bytes = np.asarray(bytearray(new_face_image.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(img_rgb)
                    
                    if face_locations:
                        face_encodings_to_save = face_recognition.face_encodings(img_rgb, face_locations)
                        
                        if face_encodings_to_save:
                            # Convert NumPy array to a list, then to a JSON string for Firestore
                            encodings_as_json_strings = [json.dumps(enc.tolist()) for enc in face_encodings_to_save]

                            # Add data to Firestore
                            doc_ref = db.collection(FIRESTORE_COLLECTION_PATH).add({
                                "name": new_face_name,
                                "encodings": encodings_as_json_strings,
                                "added_by_user_id": st.session_state.user_id, # Record who added it
                                "timestamp": firestore.SERVER_TIMESTAMP # Firestore server timestamp
                            })
                            st.success(f"Added '{new_face_name}' to database with ID: {doc_ref[1].id}")

                            # Clear the cache for load_known_faces_from_firestore to force a reload
                            load_known_faces_from_firestore.clear()
                            
                            # Re-load known faces from the actual Firestore database
                            global known_face_encodings, known_face_names # Declare global to update module-level lists
                            known_face_encodings, known_face_names = load_known_faces_from_firestore(_=np.random.rand())
                            
                            st.rerun() # Rerun to refresh the UI and known faces list
                        else:
                            st.error("Could not generate face encodings from the uploaded image.")
                    else:
                        st.error(f"No face found in the uploaded image for '{new_face_name}'. Please upload an image with a clear face.")

                except Exception as e:
                    st.error(f"Error processing image for '{new_face_name}': {e}")
                    st.error("Please ensure the image is valid and contains a clear face.")

            else:
                st.warning("Please provide both a name and upload an image.")

        st.subheader("Current Known Faces in Database 📋")
        # Display current known faces from the global lists
        if known_face_names:
            for name in sorted(set(known_face_names)): 
                st.write(f"- **{name}**")
        else:
            st.info("No faces currently registered in the database.")


    else:
        if admin_password: # Only show error if user actually typed something
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤️ using `face_recognition`, `OpenCV`, `Streamlit`, and powered by `Firebase Firestore`.")
