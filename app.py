import streamlit as st
import face_recognition
import os
import numpy as np
import cv2
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import io
import json # Required for parsing the service account JSON string

# --- Firebase Initialization (Global, using st.session_state for persistence) ---
# Use st.session_state to store Firebase client objects (db and bucket)
# This is the recommended way to handle global, persistent objects in Streamlit.
if 'db' not in st.session_state or 'bucket' not in st.session_state:
    try:
        # Load Firebase credentials from Streamlit secrets.
        firebase_credentials_dict = json.loads(st.secrets["firebase"]["service_account_json"])
        
        # Initialize Firebase Admin SDK only if it hasn't been initialized globally.
        # This check prevents "ValueError: The default Firebase app already exists" on reruns.
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_credentials_dict)
            firebase_admin.initialize_app(cred, {
                'storageBucket': st.secrets["firebase"]["storage_bucket"] 
            })
        
        # Store Firestore and Storage client instances in session state.
        st.session_state.db = firestore.client()
        st.session_state.bucket = storage.bucket()
        
        st.success("Firebase initialized successfully! 🚀")
        
    except Exception as e:
        # If initialization fails, display an error and stop the app.
        st.error(f"Error initializing Firebase: {e}. Please check your .streamlit/secrets.toml and Firebase setup carefully.")
        st.stop() # Halts script execution

# Define Firestore collection name and Storage folder from secrets for consistency.
FIRESTORE_COLLECTION_NAME = st.secrets["firebase"]["firestore_collection"]
STORAGE_KNOWN_FACES_FOLDER = "known_faces_images"

# --- Debugging line to display the bucket name being used ---
st.sidebar.info(f"App is attempting to use Storage Bucket: **`{st.secrets['firebase']['storage_bucket']}`**")


# --- Data Loading Function (Cached for performance) ---
# This function now directly accesses st.session_state.db,
# so the unhashable db_client is not passed as an argument.
@st.cache_resource(ttl=3600) 
def load_known_faces_from_firebase(_=None): 
    st.info("Loading known faces from Firebase... This might take a moment.")
    
    known_face_encodings_local = []
    known_face_names_local = []

    try:
        # Fetch all documents from the specified Firestore collection.
        # Each document is assumed to be a single face entry.
        docs = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).stream()
        for doc in docs:
            face_data = doc.to_dict()
            name = face_data.get("name")
            encoding_list = face_data.get("encoding") # Encoding is stored as a Python list of floats
            
            # Validate data and append to local lists.
            if name and encoding_list:
                known_face_encodings_local.append(np.array(encoding_list)) # Convert list back to NumPy array
                known_face_names_local.append(name)
            else:
                st.warning(f"Skipping malformed face data in Firestore document {doc.id}. Missing name or encoding.")
        
        st.success(f"Finished loading known faces. Total known faces: {len(known_face_encodings_local)}")
        return known_face_encodings_local, known_face_names_local

    except Exception as e:
        # Handle errors during data loading (e.g., network issues, permission errors).
        st.error(f"Error loading known faces from Firebase: {e}. "
                 "Ensure your Firestore collection exists and security rules are correct.")
        return [], [] # Return empty lists on error to prevent further issues

# Load known faces when the script runs. This will use the cache if available.
known_face_encodings, known_face_names = load_known_faces_from_firebase()

# --- Face Processing and Drawing Function ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    """
    Detects faces in an image, compares them to known faces, and draws bounding boxes
    and labels with dynamic text sizing.
    
    Args:
        frame_rgb (numpy.array): The input image frame in RGB format.
        known_encodings (list): List of known face encodings.
        known_names (list): List of names corresponding to known encodings.
        
    Returns:
        numpy.array: The image frame with detected faces, boxes, and labels drawn.
    """
    frame_rgb_copy = np.copy(frame_rgb)
    
    # Find all face locations and face encodings in the current frame.
    face_locations = face_recognition.face_locations(frame_rgb_copy)
    face_encodings = face_recognition.face_encodings(frame_rgb_copy, face_locations)

    # Convert the RGB frame to BGR for OpenCV drawing functions.
    frame_bgr = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)

    # If no faces are found, return the original frame.
    if not face_locations:
        return frame_bgr

    # Iterate through each detected face.
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown" # Default name for unrecognized faces

        # Only attempt to compare if there are known faces loaded.
        if known_encodings: 
            # Compare the detected face with all known faces.
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            # Calculate the distance to each known face (lower distance means better match).
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the best match (minimum distance).
            best_match_index = np.argmin(face_distances)
            
            # If the best match is actually a match (within a threshold).
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                # If not a direct match, but close enough, suggest a "Possibly" match.
                # The threshold (0.6) can be adjusted based on desired strictness.
                if face_distances[best_match_index] < 0.6: 
                    name = f"Possibly {known_names[best_match_index]}" 
                else:
                    name = "Unknown" # Clearly unknown if not close to any known face

        # --- Drawing Bounding Boxes and Labels ---
        box_padding = 15 # Padding around the face box
        base_label_height = 25 # Minimum height for the name label
        text_y_offset = 10 # Vertical offset for text within the label
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1

        # Calculate extended box coordinates with padding, ensuring they stay within image bounds.
        top_ext = max(0, top - box_padding)
        right_ext = min(frame_bgr.shape[1], right + box_padding)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding)
        left_ext = max(0, left - box_padding)

        # Draw the main bounding box around the face.
        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2)

        # Calculate text size to dynamically size the label background.
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        label_width = text_width + (box_padding * 2)
        label_height = max(base_label_height, text_height + (text_y_offset * 2))

        # Position the label background directly below the bounding box.
        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width) # Ensure label is at least as wide as box

        # Adjust label position if it goes out of image bounds.
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0: # If label goes above the top of the image
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width)

        # Draw the filled rectangle for the label background.
        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        # Calculate text position to center it within the label background.
        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        # Put the name text on the label.
        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")

# Initialize session state for page navigation.
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Home Page ---
if st.session_state.page == 'home':
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        try:
            st.image("sso_logo.jpg", width=300) 
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found. Please ensure it's in the same directory.")
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
        st.info("No known faces loaded from Firebase. Please ensure faces are added via the Admin panel.")

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

# --- Admin Login Page (Original Version) ---
elif st.session_state.page == 'admin_login':
    st.title("Admin Panel 🔒")
    st.markdown("This section is for **administrators** only.")

    admin_password_from_secrets = st.secrets["admin"]["password"]
    admin_password_input = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")

    if admin_password_input == admin_password_from_secrets:
        st.success("Welcome, Admin! 🎉")

        st.subheader("Add New Face to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        # Removed age and height inputs
        
        new_face_image = st.file_uploader("Upload Image of New Face:", 
                                             type=["jpg", "jpeg", "png"], 
                                             key="new_face_image_uploader") # Removed accept_multiple_files=True

        if st.button("Add Face to Database", key="add_face_btn"):
            if new_face_name and new_face_image:
                with st.spinner(f"Adding '{new_face_name}' to Firebase..."):
                    try:
                        # 1. Process the uploaded image to get face encodings.
                        img = Image.open(new_face_image).convert("RGB")
                        img_array = np.array(img)
                        
                        face_locations = face_recognition.face_locations(img_array)
                        face_encodings = face_recognition.face_encodings(img_array, face_locations)

                        if face_encodings:
                            # 2. Upload the image to Firebase Cloud Storage.
                            # Create a unique filename to avoid collisions.
                            unique_filename = f"{new_face_name.replace(' ', '_').lower()}_{os.urandom(4).hex()}.jpg"
                            storage_path = f"{STORAGE_KNOWN_FACES_FOLDER}/{unique_filename}"
                            
                            # Convert PIL Image to bytes for uploading to Storage.
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='JPEG')
                            img_byte_arr = img_byte_arr.getvalue()

                            # Get a blob (reference) to the file in Storage and upload.
                            blob = st.session_state.bucket.blob(storage_path)
                            blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
                            st.info(f"Image uploaded to Storage: {storage_path}")

                            # 3. Save face encoding and metadata to Cloud Firestore.
                            # Convert NumPy array encoding to a Python list for Firestore compatibility.
                            face_encoding_list = face_encodings[0].tolist() 
                            
                            # Add a new document to the Firestore collection with an auto-generated ID.
                            # Each document represents a single face entry.
                            doc_ref = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).document() 
                            doc_ref.set({
                                "name": new_face_name,
                                "encoding": face_encoding_list,
                                "image_storage_path": storage_path, # Store path for future reference
                                "timestamp": firestore.SERVER_TIMESTAMP # Optional: record the time the face was added
                            })

                            # 4. Invalidate the cache and reload known faces.
                            load_known_faces_from_firebase.clear()
                            known_face_encodings, known_face_names = load_known_faces_from_firebase(_=np.random.rand())
                            
                            st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                            st.balloons()
                            st.rerun() # Rerun to refresh the UI and ensure new faces are recognized immediately

                        else:
                            st.error(f"No face found in the uploaded image for '{new_face_name}'. Please upload an image with a clear face.")
                            
                    except Exception as e:
                        st.error(f"Error adding face to Firebase: {e}. "
                                 "Check Firebase security rules and network connection.")
            else:
                st.warning("Please provide both a name and upload an image.")

        st.subheader("Current Known Faces 📋")
        # Display the names of currently registered faces (simple list).
        if known_face_names:
            unique_names = sorted(list(set(known_face_names))) # Get unique names and sort them
            for name in unique_names:
                st.write(f"- **{name}**")
        else:
            st.info("No faces currently registered in the database.")
    else:
        if admin_password_input:
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤️ using `face_recognition`, `OpenCV`, `Streamlit`, and `Firebase`.")
