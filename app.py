import streamlit as st
import face_recognition
import os
import numpy as np
import cv2
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
from firebase_admin import exceptions as firebase_exceptions # Import Firebase exceptions
import io
import json # Required for parsing the service account JSON string
import uuid # For generating unique IDs for images

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

# --- Data Loading Function (Cached for performance) ---
# This function now directly accesses st.session_state.db,
# so the unhashable db_client is not passed as an argument.
@st.cache_resource(ttl=3600) 
def load_known_faces_from_firebase(_=None): 
    st.info("Loading known faces from Firebase... This might take a moment.")
    
    all_known_face_encodings = []
    all_known_face_names = []

    try:
        # Fetch all documents from the specified Firestore collection (each document is a person).
        person_docs = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).stream()
        
        for doc in person_docs:
            person_data = doc.to_dict()
            person_name = person_data.get("name")
            person_images = person_data.get("images", []) # Get the list of images for this person

            if person_name and person_images:
                for img_entry in person_images:
                    encoding_list = img_entry.get("encoding")
                    if encoding_list:
                        all_known_face_encodings.append(np.array(encoding_list))
                        all_known_face_names.append(person_name)
                    else:
                        st.warning(f"Skipping malformed image entry for '{person_name}' in Firestore document {doc.id}. Missing encoding.")
            else:
                st.warning(f"Skipping malformed person data in Firestore document {doc.id}. Missing name or images.")
        
        st.success(f"Finished loading known faces. Total encodings loaded: {len(all_known_face_encodings)}")
        return all_known_face_encodings, all_known_face_names

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
        known_encodings (list): List of all known face encodings from all persons.
        known_names (list): List of names corresponding to each known encoding.
        
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

# --- Admin Login Page ---
elif st.session_state.page == 'admin_login':
    st.title("Admin Panel 🔒")
    st.markdown("This section is for **administrators** only.")

    admin_password_from_secrets = st.secrets["admin"]["password"]
    admin_password_input = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")

    if admin_password_input == admin_password_from_secrets:
        st.success("Welcome, Admin! 🎉")

        st.subheader("Add New Person to Database ➕")
        st.markdown("Upload one or more pictures of a person and provide their details.")

        new_person_name = st.text_input("Enter Person's Name:", key="new_person_name_input")
        new_person_age = st.number_input("Enter Person's Age (optional):", min_value=0, max_value=150, value=None, format="%d", key="new_person_age_input")
        new_person_height = st.number_input("Enter Person's Height in cm (optional):", min_value=0, max_value=300, value=None, format="%d", key="new_person_height_input")
        
        new_person_images = st.file_uploader("Upload One or More Images of the Person:", 
                                             type=["jpg", "jpeg", "png"], 
                                             accept_multiple_files=True, 
                                             key="new_person_images_uploader")

        if st.button("Add/Update Person in Database", key="add_person_btn"):
            if new_person_name and new_person_images:
                with st.spinner(f"Processing '{new_person_name}' and uploading images to Firebase..."):
                    try:
                        person_ref = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).where("name", "==", new_person_name).limit(1).get()
                        person_doc = None
                        person_doc_id = None
                        
                        if person_ref:
                            for doc in person_ref: # Iterate to get the single document
                                person_doc = doc
                                person_doc_id = doc.id
                                break # Found the existing document

                        new_image_entries = []
                        faces_found_count = 0

                        for uploaded_file in new_person_images:
                            img = Image.open(uploaded_file).convert("RGB")
                            img_array = np.array(img)
                            
                            face_locations = face_recognition.face_locations(img_array)
                            face_encodings = face_recognition.face_encodings(img_array, face_locations)

                            if face_encodings:
                                faces_found_count += 1
                                # Generate a unique ID for each image to prevent collisions
                                image_uuid = str(uuid.uuid4())
                                
                                # If person exists, use their doc ID for storage path, otherwise a placeholder for now
                                # The actual person_doc_id will be available after the first creation or from existing.
                                temp_person_folder_id = person_doc_id if person_doc_id else "temp_person_" + str(uuid.uuid4())
                                unique_filename = f"{new_person_name.replace(' ', '_').lower()}_{image_uuid}.jpg"
                                storage_path = f"{STORAGE_KNOWN_FACES_FOLDER}/{temp_person_folder_id}/{unique_filename}"
                                
                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='JPEG')
                                img_byte_arr = img_byte_arr.getvalue()

                                blob = st.session_state.bucket.blob(storage_path)
                                blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
                                st.info(f"Image uploaded to Storage: {storage_path}")

                                face_encoding_list = face_encodings[0].tolist() 
                                
                                new_image_entries.append({
                                    "encoding": face_encoding_list,
                                    "storage_path": storage_path,
                                    "upload_timestamp": firestore.SERVER_TIMESTAMP 
                                })
                            else:
                                st.warning(f"No face found in image '{uploaded_file.name}'. Skipping this image.")
                        
                        if faces_found_count == 0:
                            st.error("No faces were found in any of the uploaded images. Please upload images with clear faces.")
                            st.stop() # Stop execution if no faces were found in any image

                        if person_doc:
                            # Person exists, update their document
                            current_images = person_doc.get("images", [])
                            updated_images = current_images + new_image_entries
                            
                            st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).document(person_doc_id).update({
                                "images": updated_images,
                                "age": new_person_age, # Update age/height even if adding more images
                                "height": new_person_height
                            })
                            st.success(f"Successfully added {faces_found_count} new image(s) for '{new_person_name}'! ✅")
                        else:
                            # Person does not exist, create a new document
                            new_doc_data = {
                                "name": new_person_name,
                                "age": new_person_age,
                                "height": new_person_height,
                                "images": new_image_entries,
                                "created_at": firestore.SERVER_TIMESTAMP
                            }
                            doc_ref = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).add(new_doc_data)
                            new_person_doc_id = doc_ref[1].id # Get the ID of the newly created document
                            
                            # Now, update the storage paths with the actual person_doc_id
                            # This is a more complex step if you want to rename files in storage.
                            # For simplicity, we'll assume the initial temp_person_folder_id is fine,
                            # or you can implement a storage move/rename here if critical.
                            # For now, the path stored in Firestore will correctly reference the image.

                            st.success(f"Successfully added new person '{new_person_name}' with {faces_found_count} image(s)! ✅")
                        
                        st.balloons()
                        load_known_faces_from_firebase.clear() # Clear cache to force reload
                        # Reload global known faces after update
                        known_face_encodings, known_face_names = load_known_faces_from_firebase(_=np.random.rand())
                        st.rerun() # Rerun to refresh the UI and known faces list

                    except firebase_exceptions.FirebaseError as fe:
                        st.error(f"Firebase Error: {fe}. Check your Firebase permissions or data structure.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please provide a name and upload at least one image.")

        st.subheader("Current Registered People 📋")
        # Fetch and display current registered people and their image counts
        try:
            current_people_docs = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).stream()
            people_list = []
            for doc in current_people_docs:
                person_data = doc.to_dict()
                name = person_data.get("name", "N/A")
                age = person_data.get("age", "N/A")
                height = person_data.get("height", "N/A")
                image_count = len(person_data.get("images", []))
                people_list.append(f"- **{name}** (Age: {age}, Height: {height} cm) - {image_count} image(s)")
            
            if people_list:
                for person_info in sorted(people_list):
                    st.write(person_info)
            else:
                st.info("No people currently registered in the database.")
        except Exception as e:
            st.error(f"Error fetching registered people: {e}")

    else:
        if admin_password_input:
            st.error("Incorrect password.")

    if st.button("⬅ Back to Home", key="admin_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤️ using `face_recognition`, `OpenCV`, `Streamlit`, and `Firebase`.")
