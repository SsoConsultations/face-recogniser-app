import os
import json
import streamlit as st
import face_recognition
import numpy as np
import cv2
from PIL import Image # Import Pillow for image manipulation
import firebase_admin
from firebase_admin import credentials, firestore, storage
import logging # Import logging for more robust error handling

# --- Configuration Constants ---
# Use environment variables for sensitive data and configurations
# Default to a placeholder if not set, but warn the user
FIREBASE_STORAGE_BUCKET_NAME = os.getenv('FIREBASE_STORAGE_BUCKET_NAME', 'face-recogniser-app.appspot.com')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
if ADMIN_PASSWORD is None:
    st.warning("ADMIN_PASSWORD environment variable not set. Using 'admin123' for demonstration. Set it securely for production!")
    ADMIN_PASSWORD = "admin123" # Fallback for demonstration, DO NOT use in production

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Firebase Initialization ---
# Initialize Firebase outside of Streamlit's main execution flow if possible,
# or ensure it's only initialized once. st.singleton is not available for firebase_admin.
# A common pattern is to put it in a function decorated with st.cache_resource.
@st.cache_resource
def initialize_firebase():
    try:
        firebase_service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT')
        if not firebase_service_account_json:
            st.error("FIREBASE_SERVICE_ACCOUNT environment variable not set. Firebase cannot be initialized.")
            st.stop() # Stop the app if critical env var is missing
            
        cred = credentials.Certificate(json.loads(firebase_service_account_json))
        firebase_admin.initialize_app(cred, {
            'storageBucket': FIREBASE_STORAGE_BUCKET_NAME
        })
        logging.info("Firebase initialized successfully.")
        return firestore.client(), storage.bucket()
    except (json.JSONDecodeError, ValueError, Exception) as e:
        st.error(f"Error initializing Firebase: {e}. Please ensure FIREBASE_SERVICE_ACCOUNT is correctly set and is a valid JSON string.")
        st.stop() # Stop the app if Firebase can't be initialized

# Initialize Firebase and get clients
db, bucket = initialize_firebase()

# --- Data Storage (use st.cache_resource for efficiency) ---
@st.cache_resource
def load_known_faces_from_firestore():
    """
    Loads known face encodings and names from Firestore and Firebase Storage.
    This function is cached to avoid repeated database calls.
    """
    st.info("Loading known faces from Firestore... This might take a moment. ⏳")
    known_face_encodings_list = []
    known_face_names_list = []

    try:
        # Firestore security rules: /artifacts/{appId}/public/data/known_faces
        # Note: In Streamlit, __app_id is not automatically available like in Canvas.
        # For a truly multi-tenant Streamlit app, you'd need to manage app IDs differently.
        # For a single Streamlit deployment, 'default-app-id' is fine, or you can make it an env var.
        app_id_for_firestore = os.getenv('STREAMLIT_APP_ID', 'default-streamlit-app-id') # Use an env var or a fixed ID
        faces_ref = db.collection(f'artifacts/{app_id_for_firestore}/public/data/known_faces').stream()
        
        for face in faces_ref:
            data = face.to_dict()
            name = data.get('name')
            image_path = data.get('image_path')

            if not name or not image_path:
                logging.warning(f"Skipping malformed face entry in Firestore: {face.id} (Missing name or image_path)")
                continue

            try:
                # Load the image from Firebase Storage
                image_blob = bucket.blob(image_path)
                image_data = image_blob.download_as_bytes()
                image = face_recognition.load_image_file(image_data)
                
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings_list.append(face_encodings[0])
                    known_face_names_list.append(name)
                    logging.info(f"Loaded face for: {name}")
                else:
                    logging.warning(f"No face found in image for {name} at path: {image_path}. Skipping.")
            except Exception as e:
                logging.error(f"Error loading image or encoding face for {name} at {image_path}: {e}")
                st.warning(f"Could not load image for '{name}'. It might be corrupted or missing in storage.")
                continue

    except Exception as e:
        st.error(f"Failed to load known faces from Firestore: {e}. Check your Firestore rules and network connectivity.")
        logging.critical(f"Critical error loading known faces: {e}")
    
    st.success(f"Finished loading known faces from Firestore. Total known faces: {len(known_face_encodings_list)} ✅")
    return known_face_encodings_list, known_face_names_list

# Initialize known faces at module level
known_face_encodings = []
known_face_names = []
# Ensure this only runs once on app startup or when cache is cleared
if not st.session_state.get('faces_loaded_initial', False):
    known_face_encodings, known_face_names = load_known_faces_from_firestore()
    st.session_state['faces_loaded_initial'] = True

# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    """
    Processes an RGB image to detect faces, recognize them, and draw bounding boxes and labels.

    Args:
        frame_rgb (numpy.array): The input image frame in RGB format.
        known_encodings (list): A list of known face encodings.
        known_names (list): A list of names corresponding to known_encodings.

    Returns:
        numpy.array: The processed image frame in BGR format with drawn faces and labels.
    """
    frame_rgb_copy = np.copy(frame_rgb)

    # Convert to grayscale for face detection if desired (face_recognition handles RGB directly)
    # face_locations = face_recognition.face_locations(frame_rgb_copy, model="cnn") # 'hog' is default, 'cnn' is slower but more accurate
    face_locations = face_recognition.face_locations(frame_rgb_copy)
    face_encodings = face_recognition.face_encodings(frame_rgb_copy, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr # Return original frame if no faces found

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        color = (0, 0, 255) # Red for Unknown

        if known_encodings: # Only compare if there are known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            
            # A common threshold for face_recognition is around 0.6.
            # Lower values mean a stricter match.
            recognition_threshold = 0.6 
            
            if matches[best_match_index] and face_distances[best_match_index] < recognition_threshold:
                name = known_names[best_match_index]
                color = (0, 255, 0) # Green for Known
            # Optional: if it's close but not a perfect match, still consider it if within threshold
            elif face_distances[best_match_index] < recognition_threshold:
                name = f"Likely {known_names[best_match_index]}?" # Indicate uncertainty
                color = (0, 165, 255) # Orange for Likely
            
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

        # Draw face box
        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), color, 2)

        # Calculate label size and position
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        label_width = text_width + (box_padding * 2)
        label_height = max(base_label_height, text_height + (text_y_offset * 2))

        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width)

        # Ensure label stays within image bounds
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0: label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width)

        # Draw label background
        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), color, cv2.FILLED)

        # Draw text on label
        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness) # White text

    return frame_bgr

# --- Streamlit UI ---
st.set_page_config(page_title="SSO Face Recogniser", layout="centered", icon="🕵️‍♂️")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Home Page ---
def home_page():
    st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True) # Adjust padding
    
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        # Use a placeholder image URL or ensure 'sso_logo.jpg' is deployed with the app
        logo_path = os.path.join(os.path.dirname(__file__), "sso_logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.image("https://placehold.co/100x100/E0E0E0/333333?text=SSO", width=100) # Placeholder
            st.warning("Logo image 'sso_logo.jpg' not found. Using a placeholder.")

    with col_title:
        st.markdown("<h1 style='text-align: left; color: #333333;'>SSO Consultants Face Recogniser 🕵️‍♂️</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #555555;'>Welcome! Please choose your login type.</h3>", unsafe_allow_html=True)

    st.write("") # Add some space

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    with col_btn1:
        if st.button("👥 Login as User", key="user_login_btn", help="Proceed to face recognition for users", use_container_width=True):
            st.session_state.page = 'user_login'
            st.rerun()

    with col_btn3:
        if st.button("⚙️ Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities", use_container_width=True):
            st.session_state.page = 'admin_login'
            st.rerun()
    
    st.markdown("<div style='text-align: center; margin-top: 50px; color: #777777; font-size: 0.9em;'>Developed with ❤️ using face_recognition, OpenCV, and Streamlit.</div>", unsafe_allow_html=True)


# --- User Login (Face Recognition) Page ---
def user_login_page():
    st.title("Face Recognition & Detection 📸")
    st.markdown("""
    This application can detect and recognize faces from your live webcam or an uploaded image.
    Recognized faces will be labeled with their names, while unknown faces will be marked as "Unknown".
    """)

    if not known_face_encodings:
        st.warning("⚠️ No known faces loaded. Face recognition will only detect faces, not identify them. Please ask an admin to add faces.")

    st.subheader("Choose Input Method")
    input_method = st.radio("", ("Live Webcam", "Upload Image"), key="user_input_method", horizontal=True)

    processed_image_placeholder = st.empty() # Placeholder for the processed image

    if input_method == "Live Webcam":
        st.info("Allow camera access. Click 'Take Photo' to capture an image for processing.")
        camera_image = st.camera_input("Capture Photo:", key="user_camera_input")

        if camera_image:
            with st.spinner("Processing live image... 🔄"):
                try:
                    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                    processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
                    
                    processed_image_placeholder.image(processed_img_rgb, caption="Processed Live Image", use_column_width=True)
                    st.success("Face detection and recognition complete! ✨")
                except Exception as e:
                    st.error(f"Error processing webcam image: {e}")
                    logging.error(f"Error in webcam image processing: {e}")
        else:
            processed_image_placeholder.info("Waiting for webcam input. Click 'Capture Photo' above.")

    elif input_method == "Upload Image":
        st.info("Upload an image file (JPG, PNG, BMP, GIF) for face detection and recognition.")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file:
            with st.spinner("Loading and processing image... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # Display original image first
                    st.image(img_rgb, caption="Original Uploaded Image", use_column_width=True)

                    processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                    processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
                    
                    processed_image_placeholder.image(processed_img_rgb, caption="Processed Image with Faces", use_column_width=True)
                    st.success("Face detection and recognition complete! ✨")
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    logging.error(f"Error in uploaded image processing: {e}")
        else:
            processed_image_placeholder.info("Please upload an image file.")

    st.markdown("---")
    if st.button("⬅️ Back to Home", key="user_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Login Page ---
def admin_login_page():
    st.title("Admin Panel 🔒")
    st.markdown("This section is for **administrators** only.")

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        admin_password_input = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")
        if st.button("Login", key="admin_login_btn_submit"):
            if admin_password_input == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Welcome, Admin! 🎉")
                st.rerun() # Rerun to show admin features
            else:
                st.error("Incorrect password. ❌")
    
    if st.session_state.admin_logged_in:
        st.subheader("Add New Faces to Database ➕")
        st.markdown("Upload an image of a person and provide a name for recognition. Ensure the image clearly shows one face.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_image = st.file_uploader("Upload Image of New Face:", type=["jpg", "jpeg", "png"], key="new_face_image_uploader")

        if st.button("Add Face to Database", key="add_face_btn", use_container_width=True):
            if new_face_name and new_face_image:
                with st.spinner(f"Adding '{new_face_name}' to database... 🚀"):
                    try:
                        # Convert uploaded file to numpy array for face_recognition
                        file_bytes = np.asarray(bytearray(new_face_image.read()), dtype=np.uint8)
                        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # Check if a face is present in the new image
                        face_locations_new_face = face_recognition.face_locations(img_rgb)
                        if not face_locations_new_face:
                            st.warning("No face detected in the uploaded image. Please upload an image with a clear face.")
                            logging.warning(f"No face detected in uploaded image for {new_face_name}.")
                            st.stop() # Stop further execution for this button click

                        # Generate a more unique filename to prevent collisions
                        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                        clean_name = new_face_name.replace(' ', '_').lower().replace('/', '_').replace('\\', '_')
                        image_filename = f"known_faces/{clean_name}_{timestamp}.jpg"
                        
                        # Upload the image to Firebase Storage
                        # Reset file pointer before re-reading for upload_from_file
                        new_face_image.seek(0) 
                        blob = bucket.blob(image_filename)
                        blob.upload_from_file(new_face_image)
                        logging.info(f"Uploaded image to Firebase Storage: {image_filename}")

                        # Save the image path and name to Firestore
                        app_id_for_firestore = os.getenv('STREAMLIT_APP_ID', 'default-streamlit-app-id')
                        db.collection(f'artifacts/{app_id_for_firestore}/public/data/known_faces').add({
                            'name': new_face_name,
                            'image_path': image_filename # Store the path in Firestore
                        })
                        logging.info(f"Added entry to Firestore for: {new_face_name}")

                        # Clear the cache and reload known faces to include the new entry
                        load_known_faces_from_firestore.clear()
                        # IMPORTANT: Re-assign to the module-level variables without 'global' in this scope
                        # The variables are already global, this re-assigns their values.
                        global known_face_encodings, known_face_names # This is still needed because known_face_encodings and known_face_names are reassigned here.
                        known_face_encodings, known_face_names = load_known_faces_from_firestore()

                        st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                        st.balloons() # Visual feedback
                        st.rerun() # Rerun to update the "Current Known Faces" list
                    except Exception as e:
                        st.error(f"Error adding face: {e}")
                        logging.error(f"Error adding face for {new_face_name}: {e}")
            else:
                st.warning("Please provide both a name and upload an image. 📝")

        st.subheader("Current Known Faces 📋")
        if known_face_names:
            # Display unique names and provide delete option
            unique_names = sorted(list(set(known_face_names)))
            for name in unique_names:
                col_name, col_delete = st.columns([3, 1])
                with col_name:
                    st.write(f"- **{name}**")
                with col_delete:
                    if st.button("Delete", key=f"delete_btn_{name}"):
                        if st.warning(f"Are you sure you want to delete ALL entries for '{name}'?"):
                            # This is a simple confirmation. For critical ops, use a custom modal.
                            # In Streamlit, a simple way is to ask for re-confirmation
                            st.session_state[f'confirm_delete_{name}'] = True
                            st.rerun() # Rerun to show confirmation button
                        
                        if st.session_state.get(f'confirm_delete_{name}', False):
                            if st.button(f"Confirm Delete '{name}'", key=f"confirm_delete_btn_{name}"):
                                with st.spinner(f"Deleting all entries for '{name}'..."):
                                    try:
                                        app_id_for_firestore = os.getenv('STREAMLIT_APP_ID', 'default-streamlit-app-id')
                                        faces_to_delete_ref = db.collection(f'artifacts/{app_id_for_firestore}/public/data/known_faces').where('name', '==', name).stream()
                                        
                                        deleted_count = 0
                                        for doc_to_delete in faces_to_delete_ref:
                                            data_to_delete = doc_to_delete.to_dict()
                                            image_path_to_delete = data_to_delete.get('image_path')
                                            
                                            # Delete from Storage
                                            if image_path_to_delete:
                                                try:
                                                    blob_to_delete = bucket.blob(image_path_to_delete)
                                                    blob_to_delete.delete()
                                                    logging.info(f"Deleted image from Storage: {image_path_to_delete}")
                                                except Exception as storage_e:
                                                    logging.warning(f"Could not delete image {image_path_to_delete} from Storage: {storage_e}")
                                                    st.warning(f"Could not delete image for '{name}' from Storage.")

                                            # Delete from Firestore
                                            db.collection(f'artifacts/{app_id_for_firestore}/public/data/known_faces').document(doc_to_delete.id).delete()
                                            deleted_count += 1
                                            logging.info(f"Deleted Firestore document: {doc_to_delete.id}")
                                        
                                        load_known_faces_from_firestore.clear()
                                        global known_face_encodings, known_face_names # Re-declare global for re-assignment
                                        known_face_encodings, known_face_names = load_known_faces_from_firestore()
                                        
                                        st.success(f"Successfully deleted {deleted_count} entries for '{name}'. ✅")
                                        del st.session_state[f'confirm_delete_{name}'] # Clear confirmation state
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting entries for '{name}': {e}")
                                        logging.error(f"Error deleting entries for '{name}': {e}")
                            else:
                                st.info("Deletion cancelled.")
                                del st.session_state[f'confirm_delete_{name}'] # Clear confirmation state
                                st.rerun() # Rerun to remove confirmation button
        else:
            st.info("No faces currently registered in the database. 🤷‍♀️")

    st.markdown("---")
    if st.button("⬅️ Back to Home", key="admin_back_btn"):
        st.session_state.admin_logged_in = False # Log out admin on going back
        st.session_state.page = 'home'
        st.rerun()

# --- Main App Logic ---
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'user_login':
    user_login_page()
elif st.session_state.page == 'admin_login':
    admin_login_page()
