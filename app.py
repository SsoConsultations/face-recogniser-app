import streamlit as st
import face_recognition
import os
import numpy as np
import cv2
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import io
import json

# --- Firebase Initialization (Global, using st.session_state for persistence) ---
if 'db' not in st.session_state or 'bucket' not in st.session_state:
    try:
        firebase_credentials_dict = json.loads(st.secrets["firebase"]["service_account_json"])
        
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_credentials_dict)
            firebase_admin.initialize_app(cred, {
                'storageBucket': st.secrets["firebase"]["storage_bucket"] 
            })
        
        st.session_state.db = firestore.client()
        st.session_state.bucket = storage.bucket()
        
        # st.success("Firebase initialized successfully! 🚀") # Removed as requested
        
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}. Please check your .streamlit/secrets.toml and Firebase setup carefully.")
        st.stop()

FIRESTORE_COLLECTION_NAME = st.secrets["firebase"]["firestore_collection"]
STORAGE_KNOWN_FACES_FOLDER = "known_faces_images"

# --- Data Loading Function (Cached for performance) ---
@st.cache_resource(ttl=3600) 
def load_known_faces_from_firebase(_=None): 
    # st.info("Loading known faces from Firebase... This might take a moment.") # Removed as requested
    
    known_face_encodings_local = []
    known_face_names_local = []

    try:
        docs = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).stream()
        for doc in docs:
            face_data = doc.to_dict()
            name = face_data.get("name")
            encoding_list = face_data.get("encoding")
            
            if name and encoding_list:
                known_face_encodings_local.append(np.array(encoding_list))
                known_face_names_local.append(name)
            else:
                st.warning(f"Skipping malformed face data in Firestore document {doc.id}. Missing name or encoding.")
        
        # st.success(f"Finished loading known faces. Total known faces: {len(known_face_encodings_local)}") # Removed as requested
        return known_face_encodings_local, known_face_names_local

    except Exception as e:
        st.error(f"Error loading known faces from Firebase: {e}. "
                 "Ensure your Firestore collection exists and security rules are correct.")
        return [], []

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
    
    face_locations = face_recognition.face_locations(frame_rgb_copy)
    face_encodings = face_recognition.face_encodings(frame_rgb_copy, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)

    if not face_locations:
        return frame_bgr

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if known_encodings: 
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                if face_distances[best_match_index] < 0.6: 
                    name = f"Possibly {known_names[best_match_index]}" 
                else:
                    name = "Unknown"

        # --- Dynamic Drawing Parameters ---
        # Calculate face width and height
        face_width = right - left
        face_height = bottom - top
        
        # Base multiplier for font and thickness, adjust as needed
        # These values were chosen after testing to provide a good balance.
        # You might need to tweak `base_font_size` and `base_thickness` further.
        base_font_size = 0.002 # A factor of the face_width for font_scale
        base_thickness = 0.005 # A factor of the face_width for line thickness
        
        # Ensure minimums to prevent text/lines from disappearing on very small faces
        min_font_scale = 0.5
        min_thickness = 1
        
        font_scale = max(min_font_scale, base_font_size * face_width)
        font_thickness = max(min_thickness, int(base_thickness * face_width))
        line_thickness = max(2, int(face_width * 0.01)) # Make the box line slightly thicker than text thickness

        box_padding_factor = 0.1 # Padding as a percentage of face width/height
        box_padding_x = int(face_width * box_padding_factor)
        box_padding_y = int(face_height * box_padding_factor)

        # Calculate extended box coordinates with dynamic padding.
        # Ensure they stay within image bounds.
        top_ext = max(0, top - box_padding_y)
        right_ext = min(frame_bgr.shape[1], right + box_padding_x)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding_y)
        left_ext = max(0, left - box_padding_x)

        # Draw the main bounding box around the face with dynamic thickness.
        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), line_thickness)

        # Calculate text size using the dynamic font scale and thickness.
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        
        # Use dynamic padding for label width
        label_padding_x = int(text_width * 0.2) # Some padding relative to text width
        label_padding_y = int(text_height * 0.5) # Some padding relative to text height

        label_width = text_width + (label_padding_x * 2)
        label_height = text_height + (label_padding_y * 2) + baseline

        # Position the label background directly below the bounding box, dynamically adjusting width.
        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width) # Ensure label is at least as wide as box

        # Adjust label position if it goes out of image bounds.
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0:
                label_top = 0 # Prevent label from going completely off-screen upwards

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width)

        # Draw the filled rectangle for the label background.
        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        # Calculate text position to center it within the label background.
        text_x = label_left + label_padding_x
        text_y = label_top + label_padding_y + text_height 

        # Put the name text on the label.
        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")

# Initialize session state for page navigation and login status.
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'logged_in_as_user' not in st.session_state: # Track user login status
    st.session_state.logged_in_as_user = False
if 'logged_in_as_admin' not in st.session_state: # Track admin login status
    st.session_state.logged_in_as_admin = False


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
            st.session_state.page = 'user_auth' # Redirect to user authentication page
            st.rerun()

    with col3_btn:
        if st.button("Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities"):
            st.session_state.page = 'admin_auth' # Redirect to admin authentication page
            st.rerun()
            
# --- User Authentication Page ---
elif st.session_state.page == 'user_auth':
    st.title("User Login 👤")
    st.markdown("Please enter your **username** and **password** to proceed to face recognition.")

    user_username_input = st.text_input("Username:", key="user_username_input")
    user_password_input = st.text_input("Password:", type="password", key="user_password_input")

    if st.button("Login", key="submit_user_login"):
        # Retrieve user credentials from secrets
        user_credentials = st.secrets["users"]
        authenticated = False
        # Iterate through user secrets to find a match
        for key in user_credentials:
            if key.endswith("_username") and user_credentials[key] == user_username_input:
                password_key = key.replace("_username", "_password")
                if password_key in user_credentials and user_credentials[password_key] == user_password_input:
                    authenticated = True
                    break
        
        if authenticated:
            st.success("User login successful! Redirecting to Face Recognition... 🎉")
            st.session_state.logged_in_as_user = True
            st.session_state.page = 'user_recognition' # Navigate to the user recognition page
            st.rerun()
        else:
            st.error("Invalid username or password for user.")

    if st.button("⬅ Back to Home", key="user_auth_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- User Recognition Page (Accessible only after user login) ---
elif st.session_state.page == 'user_recognition':
    # Ensure user is logged in to access this page
    if not st.session_state.logged_in_as_user:
        st.warning("Please log in as a user to access this page.")
        st.session_state.page = 'user_auth'
        st.rerun()

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

            st.image(processed_img_rgb, caption="Processed Live Image", use_container_width=True)
            # st.success("Face detection and recognition complete!") # Removed as requested
        else:
            # st.warning("Waiting for webcam input. Click 'Take Photo' above.") # Removed as requested
            pass # Keep pass if you don't want any message

    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption="Original Uploaded Image", use_container_width=True)

                processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

            st.image(processed_img_rgb, caption="Processed Image with Faces", use_container_width=True)
            # st.success("Face detection and recognition complete!") # Removed as requested
        else:
            # st.info("Please upload an image file using the browser button.") # Removed as requested
            pass # Keep pass if you don't want any message

    if st.button("⬅ Log Out and Go Home", key="user_logout_btn"):
        st.session_state.logged_in_as_user = False
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Authentication Page ---
elif st.session_state.page == 'admin_auth':
    st.title("Admin Login 🔒")
    st.markdown("Please enter your **admin username** and **password**.")

    admin_username_input = st.text_input("Admin Username:", key="admin_username_input")
    admin_password_input = st.text_input("Admin Password:", type="password", key="admin_pass_input")

    if st.button("Login", key="submit_admin_login"):
        # Validate admin credentials against secrets
        if admin_username_input == st.secrets["admin"]["username"] and \
           admin_password_input == st.secrets["admin"]["password"]:
            st.success("Admin login successful! Redirecting to Admin Panel... 🎉")
            st.session_state.logged_in_as_admin = True
            st.session_state.page = 'admin_panel' # Navigate to the admin panel
            st.rerun()
        else:
            st.error("Invalid username or password for admin.")

    if st.button("⬅ Back to Home", key="admin_auth_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Panel (Accessible only after admin login) ---
elif st.session_state.page == 'admin_panel':
    # Ensure admin is logged in to access this page
    if not st.session_state.logged_in_as_admin:
        st.warning("Please log in as an admin to access this page.")
        st.session_state.page = 'admin_auth'
        st.rerun()

    st.title("Admin Panel 🔒")
    st.markdown("This section is for **administrators** only.")

    st.subheader("Add New Face to Database ➕")
    st.markdown("Upload an image of a person and provide a name for recognition.")

    new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
    
    new_face_image = st.file_uploader("Upload Image of New Face:", 
                                     type=["jpg", "jpeg", "png"], 
                                     key="new_face_image_uploader")

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
                        # Pass a dummy argument to force cache bust (e.g., a random number)
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

    if st.button("⬅ Log Out and Go Home", key="admin_logout_btn"):
        st.session_state.logged_in_as_admin = False
        st.session_state.page = 'home'
        st.rerun()

st.markdown("---")
st.markdown("Developed with ❤️ using `face_recognition`, `OpenCV`, `Streamlit`, and `Firebase`.")
