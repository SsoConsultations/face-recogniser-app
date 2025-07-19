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
# Check if Firebase app is already initialized to prevent re-initialization errors
if 'db' not in st.session_state or 'bucket' not in st.session_state:
    try:
        # Load Firebase service account credentials from Streamlit secrets
        firebase_credentials_dict = json.loads(st.secrets["firebase"]["service_account_json"])
        
        # Initialize Firebase app only if it hasn't been initialized before
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_credentials_dict)
            firebase_admin.initialize_app(cred, {
                'storageBucket': st.secrets["firebase"]["storage_bucket"] 
            })
        
        # Store Firestore client and Storage bucket in session state for easy access
        st.session_state.db = firestore.client()
        st.session_state.bucket = storage.bucket()
        
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}. Please check your .streamlit/secrets.toml and Firebase setup carefully.")
        st.stop() # Stop the app if Firebase initialization fails

# Define Firestore collection name and Storage folder name from Streamlit secrets
FIRESTORE_COLLECTION_NAME = st.secrets["firebase"]["firestore_collection"]
STORAGE_KNOWN_FACES_FOLDER = "known_faces_images"

# --- Data Loading Function (Cached for performance) ---
@st.cache_resource(ttl=3600) # Cache the resource for 1 hour to avoid frequent Firebase calls
def load_known_faces_from_firebase(_=None): 
    """
    Loads known face encodings, names, and additional details from Firestore.
    The '_' parameter is a dummy to allow clearing the cache.
    """
    known_face_encodings_local = []
    known_face_names_local = []
    known_face_details_local = [] # List to store dictionaries of all details

    try:
        # Stream documents from the specified Firestore collection
        docs = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).stream()
        for doc in docs:
            face_data = doc.to_dict()
            name = face_data.get("name")
            encoding_list = face_data.get("encoding")
            age = face_data.get("age")
            height = face_data.get("height")
            
            if name and encoding_list:
                known_face_encodings_local.append(np.array(encoding_list))
                known_face_names_local.append(name)
                # Store all relevant details as a dictionary
                known_face_details_local.append({"name": name, "age": age, "height": height})
            else:
                st.warning(f"Skipping malformed face data in Firestore document {doc.id}. Missing name or encoding.")
        
        return known_face_encodings_local, known_face_names_local, known_face_details_local

    except Exception as e:
        st.error(f"Error loading known faces from Firebase: {e}. "
                    "Ensure your Firestore collection exists and security rules are correct.")
        return [], [], [] # Return empty lists on error

# Load known faces at the start of the app
known_face_encodings, known_face_names, known_face_details = load_known_faces_from_firebase()

# --- Face Processing and Drawing Function ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names, known_details):
    """
    Detects faces in an image, compares them to known faces, and draws bounding boxes
    and labels with only the name on the image. Other details are returned for sidebar display.
    
    Args:
        frame_rgb (numpy.array): The input image frame in RGB format.
        known_encodings (list): List of known face encodings.
        known_names (list): List of names corresponding to known encodings.
        known_details (list): List of dictionaries containing all known face details (name, age, height).
        
    Returns:
        numpy.array: The image frame with detected faces, boxes, and only the name label drawn.
        list: A list of dictionaries, each containing details for a detected and identified face.
    """
    frame_rgb_copy = np.copy(frame_rgb)
    
    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame_rgb_copy)
    face_encodings = face_recognition.face_encodings(frame_rgb_copy, face_locations)

    # Convert the image back to BGR for OpenCV drawing functions
    frame_bgr = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)

    detected_face_info = [] # List to store details for sidebar display

    # If no faces are found, clear any previously detected face info in session state
    # and return the original frame and empty list.
    if not face_locations:
        if 'detected_faces_sidebar_info' in st.session_state:
            del st.session_state['detected_faces_sidebar_info']
        return frame_bgr, []

    # Iterate through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name_on_image = "Unknown" # Default label for the image
        current_face_details = {"name": "Unknown", "age": "N/A", "height": "N/A"} # Default details for sidebar
        
        if known_encodings: # Only compare if there are known faces in the database
            # Compare current face with all known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the best match (smallest distance)
            best_match_index = np.argmin(face_distances)
            
            # If a definite match is found
            if matches[best_match_index]:
                matched_person_details = known_details[best_match_index]
                name = matched_person_details.get("name", "N/A")
                age = matched_person_details.get("age", "N/A")
                height = matched_person_details.get("height", "N/A")
                
                name_on_image = f"Name: {name}" # Label for image
                current_face_details = {"name": name, "age": age, "height": height} # Details for sidebar
            else:
                # If no definite match, but a close enough match exists (threshold 0.6)
                if face_distances[best_match_index] < 0.6: 
                    matched_person_details = known_details[best_match_index]
                    name = matched_person_details.get("name", "N/A")
                    age = matched_person_details.get("age", "N/A")
                    height = matched_person_details.get("height", "N/A")
                    
                    name_on_image = f"Possibly {name}" # Label for image
                    current_face_details = {"name": f"Possibly {name}", "age": age, "height": height} # Details for sidebar
                else:
                    name_on_image = "Unknown" # No close match
                    current_face_details = {"name": "Unknown", "age": "N/A", "height": "N/A"}

        # Add the details of the current detected face to the list for sidebar
        detected_face_info.append(current_face_details)

        # --- Dynamic Drawing Parameters for image labels ---
        # Calculate dimensions for dynamic text sizing
        face_width = right - left
        face_height = bottom - top
        
        # Base font size and thickness, scaled by face width for responsiveness
        base_font_size = 0.002 
        base_thickness = 0.005 
        
        # Minimum font scale and thickness to ensure readability on small faces
        min_font_scale = 0.5
        min_thickness = 1
        
        font_scale = max(min_font_scale, base_font_size * face_width)
        font_thickness = max(min_thickness, int(base_thickness * face_width))
        line_thickness = max(2, int(face_width * 0.01)) # Thickness for the bounding box

        # Padding around the face box
        box_padding_factor = 0.1 
        box_padding_x = int(face_width * box_padding_factor)
        box_padding_y = int(face_height * box_padding_factor)

        # Extend the bounding box with padding, ensuring it stays within image bounds
        top_ext = max(0, top - box_padding_y)
        right_ext = min(frame_bgr.shape[1], right + box_padding_x)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding_y)
        left_ext = max(0, left - box_padding_x)

        # Draw the bounding box around the face
        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), line_thickness)

        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Calculate size needed for the name label only
        (text_width, text_height), baseline = cv2.getTextSize(name_on_image, font, font_scale, font_thickness)
        
        # Padding for the label background
        label_padding_x = int(text_width * 0.1) 
        label_padding_y = int(text_height * 0.3) 

        # Calculate label background dimensions
        label_width = text_width + (label_padding_x * 2)
        label_height = text_height + (label_padding_y * 2)

        # Position the label background below the face box
        label_top = bottom_ext
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width) 

        # Adjust label position if it goes out of image bounds
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height
            if label_top < 0:
                label_top = 0

        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width)

        # Draw the filled rectangle for the label background
        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        # Draw the name text on the label background
        text_x = label_left + label_padding_x
        text_y = int(label_top + label_padding_y + text_height) # Position text correctly within the label box
        cv2.putText(frame_bgr, name_on_image, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr, detected_face_info

# --- Streamlit UI Layout ---
# Set page configuration: title, layout (centered), and initial sidebar state (expanded)
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered", initial_sidebar_state="expanded") 

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'logged_in_as_user' not in st.session_state: 
    st.session_state.logged_in_as_user = False
if 'logged_in_as_admin' not in st.session_state: 
    st.session_state.logged_in_as_admin = False
# Session state to store detected face details for sidebar display
if 'detected_faces_sidebar_info' not in st.session_state:
    st.session_state.detected_faces_sidebar_info = []


# --- Home Page ---
if st.session_state.page == 'home':
    
    # Display SSO Consultants logo or a warning if not found
    try:
        st.image("sso_logo.jpg", width=150) 
    except FileNotFoundError:
        st.warning("Logo image 'sso_logo.jpg' not found. Please ensure it's in the same directory.")
        st.markdown("## SSO Consultants") 

    st.markdown("<h3 style='margin-bottom: 0px;'>SSO Consultants Face Recogniser 🕵️‍♂️</h3>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top: 5px; margin-bottom: 20px; font-size:1.1em;'>Please choose your login type.</p>", unsafe_allow_html=True)

    # Create two columns for login buttons
    col1_btn, col2_btn = st.columns([0.2, 0.2]) 

    with col1_btn:
        if st.button("Login as User", key="user_login_btn", help="Proceed to face recognition for users"):
            st.session_state.page = 'user_auth' # Change page to user authentication
            st.rerun() # Rerun the app to switch page

    with col2_btn:
        if st.button("Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities"):
            st.session_state.page = 'admin_auth' # Change page to admin authentication
            st.rerun() # Rerun the app to switch page


# --- User Authentication Page ---
elif st.session_state.page == 'user_auth':
    st.title("User Login")
    st.markdown("Please enter your **username** and **password** to proceed to face recognition.")

    user_username_input = st.text_input("Username:", key="user_username_input")
    user_password_input = st.text_input("Password:", type="password", key="user_password_input")

    if st.button("Login", key="submit_user_login"):
        user_credentials = st.secrets["users"] # Load user credentials from secrets
        authenticated = False
        # Iterate through user credentials to find a match
        for key in user_credentials:
            if key.endswith("_username") and user_credentials[key] == user_username_input:
                password_key = key.replace("_username", "_password")
                if password_key in user_credentials and user_credentials[password_key] == user_password_input:
                    authenticated = True
                    break
        
        if authenticated:
            st.success("User login successful! Redirecting to Face Recognition...")
            st.session_state.logged_in_as_user = True # Set user login status
            st.session_state.page = 'user_recognition' # Change page to user recognition
            st.rerun()
        else:
            st.error("Invalid username or password for user.")

    if st.button("Back to Home", key="user_auth_back_btn"):
        st.session_state.page = 'home' # Go back to home page
        st.rerun()

# --- User Recognition Page (Accessible only after user login) ---
elif st.session_state.page == 'user_recognition':
    # Redirect to user login if not authenticated
    if not st.session_state.logged_in_as_user:
        st.warning("Please log in as a user to access this page.")
        st.session_state.page = 'user_auth'
        st.rerun()

    st.title("Face Recognition App with Dynamic Labels")
    st.markdown("""
    This application performs face recognition from your live webcam or an uploaded image.
    Recognized names appear on the image, while other details are shown in the sidebar.
    """)

    if not known_face_encodings:
        st.info("No known faces loaded from Firebase. Please ensure faces are added via the Admin panel.")

    # Sidebar content for the user recognition page
    with st.sidebar:
        try:
            st.image("sso_logo.jpg", width=150) # Logo in sidebar
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found in sidebar.")
        st.markdown("---")
        st.header("Choose Input Method")
        option = st.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"), key="user_input_option_sidebar")
        
        st.markdown("---")
        st.header("Recognized Faces Details") # Section for displaying recognized face details
        
        # Display details from session state if available
        if st.session_state.detected_faces_sidebar_info:
            for i, face_detail in enumerate(st.session_state.detected_faces_sidebar_info):
                # Display name prominently as a sub-heading
                st.markdown(f"### Name: **{face_detail.get('name', 'Unknown')}**") 
                if face_detail.get('age') != "N/A":
                    st.write(f"**Age:** {face_detail.get('age', 'N/A')}")
                if face_detail.get('height') != "N/A":
                    st.write(f"**Height:** {face_detail.get('height', 'N/A')}")
        else:
            st.info("No faces detected or recognized yet.")

        st.markdown("---")
        # Logout button in sidebar
        if st.button("Log Out", key="user_logout_sidebar_btn"): 
            st.session_state.logged_in_as_user = False
            # Clear detected faces info on logout
            if 'detected_faces_sidebar_info' in st.session_state:
                del st.session_state['detected_faces_sidebar_info']
            st.session_state.page = 'home' # Go back to home page
            st.rerun()

    if option == "Live Webcam Recognition":
        st.subheader("Live Webcam Face Recognition")

        camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="user_camera_input")

        # Process the image if a photo is taken
        if camera_image is not None:
            with st.spinner("Processing live image..."):
                # Read image bytes and decode using OpenCV
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Process the frame for faces and get both the image and details
                processed_img_bgr, detected_faces_info = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names, known_face_details)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                # Store detected face details in session state for sidebar display
                st.session_state['detected_faces_sidebar_info'] = detected_faces_info

            st.image(processed_img_rgb, caption="Processed Live Image", use_container_width=True)
            st.rerun() # Rerun to update the sidebar with new information
        else:
            # Clear sidebar info if no image is currently present in the camera input
            if 'detected_faces_sidebar_info' in st.session_state:
                del st.session_state['detected_faces_sidebar_info']
            
            # This logic ensures the sidebar clears if the camera input goes from having an image to being empty
            if 'prev_camera_image_state' not in st.session_state:
                st.session_state.prev_camera_image_state = None

            if st.session_state.prev_camera_image_state is not None and camera_image is None:
                st.rerun() # Trigger a rerun to clear the sidebar
            st.session_state.prev_camera_image_state = camera_image


    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        # Process the uploaded file if present
        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                # Read image bytes and decode using OpenCV
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption="Original Uploaded Image", use_container_width=True)

                # Process the frame for faces and get both the image and details
                processed_img_bgr, detected_faces_info = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names, known_face_details)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                # Store detected face details in session state for sidebar display
                st.session_state['detected_faces_sidebar_info'] = detected_faces_info

            st.image(processed_img_rgb, caption="Processed Image with Faces", use_container_width=True)
            st.rerun() # Rerun to update the sidebar with new information
        else:
            # Clear sidebar info if no file is uploaded
            if 'detected_faces_sidebar_info' in st.session_state:
                del st.session_state['detected_faces_sidebar_info']
            
            # This logic ensures the sidebar clears if the file uploader goes from having a file to being empty
            if 'prev_uploaded_file_state' not in st.session_state:
                st.session_state.prev_uploaded_file_state = None

            if st.session_state.prev_uploaded_file_state is not None and uploaded_file is None:
                st.rerun() # Trigger a rerun to clear the sidebar
            st.session_state.prev_uploaded_file_state = uploaded_file

# --- Admin Authentication Page ---
elif st.session_state.page == 'admin_auth':
    st.title("Admin Login")
    st.markdown("Please enter your **admin username** and **password**.")

    admin_username_input = st.text_input("Admin Username:", key="admin_username_input")
    admin_password_input = st.text_input("Admin Password:", type="password", key="admin_pass_input")

    if st.button("Login", key="submit_admin_login"):
        # Authenticate admin using credentials from Streamlit secrets
        if admin_username_input == st.secrets["admin"]["username"] and \
           admin_password_input == st.secrets["admin"]["password"]:
            st.success("Admin login successful! Redirecting to Admin Panel...")
            st.session_state.logged_in_as_admin = True # Set admin login status
            st.session_state.page = 'admin_panel' # Change page to admin panel
            st.rerun()
        else:
            st.error("Invalid username or password for admin.")

    if st.button("Back to Home", key="admin_auth_back_btn"):
        st.session_state.page = 'home' # Go back to home page
        st.rerun()

# --- Admin Panel (Accessible only after admin login) ---
elif st.session_state.page == 'admin_panel':
    # Redirect to admin login if not authenticated
    if not st.session_state.logged_in_as_admin:
        st.warning("Please log in as an admin to access this page.")
        st.session_state.page = 'admin_auth'
        st.rerun()

    # Sidebar content for the admin panel
    with st.sidebar:
        try:
            st.image("sso_logo.jpg", width=150) # Logo in sidebar
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found in sidebar.")
        st.markdown("---")
        st.header("Faces in Database")
        # Display list of known faces
        if known_face_names:
            for name in sorted(list(set(known_face_names))):
                st.write(f"- {name}")
        else:
            st.info("No faces currently registered.")
        st.markdown("---")
        # Logout button in sidebar
        if st.button("Log Out", key="admin_logout_sidebar_btn"): 
            st.session_state.logged_in_as_admin = False
            st.session_state.page = 'home' # Go back to home page
            st.rerun()

    st.title("Admin Panel")
    st.markdown("This section is for **administrators** only.")

    st.subheader("Add New Face to Database")
    st.markdown("Upload an image of a person and provide a name and details for recognition.")

    # Input fields for new face details
    new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
    new_face_age = st.number_input("Enter Age (optional):", min_value=0, max_value=150, value=None, format="%d", key="new_face_age_input")
    new_face_height = st.text_input("Enter Height (e.g., 5'10\" or 178cm) (optional):", key="new_face_height_input")
    
    new_face_image = st.file_uploader("Upload Image of New Face:", 
                                     type=["jpg", "jpeg", "png"], 
                                     key="new_face_image_uploader")

    # Button to add face to database
    if st.button("Add Face to Database", key="add_face_btn"):
        if new_face_name and new_face_image:
            with st.spinner(f"Adding '{new_face_name}' to Firebase..."):
                try:
                    # Open and convert image to RGB numpy array
                    img = Image.open(new_face_image).convert("RGB")
                    img_array = np.array(img)
                    
                    # Find face locations and encodings in the uploaded image
                    face_locations = face_recognition.face_locations(img_array)
                    face_encodings = face_recognition.face_encodings(img_array, face_locations)

                    if face_encodings:
                        # Generate a unique filename for storage
                        unique_filename = f"{new_face_name.replace(' ', '_').lower()}_{os.urandom(4).hex()}.jpg"
                        storage_path = f"{STORAGE_KNOWN_FACES_FOLDER}/{unique_filename}"
                        
                        # Convert image to bytes for uploading to Firebase Storage
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()

                        # Upload image to Firebase Storage
                        blob = st.session_state.bucket.blob(storage_path)
                        blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
                        st.info(f"Image uploaded to Storage: {storage_path}")

                        # Convert face encoding to a list for Firestore (NumPy arrays are not directly supported)
                        face_encoding_list = face_encodings[0].tolist() 
                        
                        # Create a new document reference in Firestore
                        doc_ref = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).document() 
                        doc_data = {
                            "name": new_face_name,
                            "encoding": face_encoding_list,
                            "image_storage_path": storage_path,
                            "timestamp": firestore.SERVER_TIMESTAMP # Add a server timestamp
                        }
                        # Add optional fields if provided
                        if new_face_age is not None:
                            doc_data["age"] = new_face_age
                        if new_face_height:
                            doc_data["height"] = new_face_height

                        # Set the document in Firestore
                        doc_ref.set(doc_data)

                        # Clear the cache for known faces so it reloads from Firebase
                        load_known_faces_from_firebase.clear()
                        # Reload known faces to update the sidebar display immediately
                        known_face_encodings, known_face_names, known_face_details = load_known_faces_from_firebase(_=np.random.rand()) 
                        
                        st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                        st.balloons() # Celebrate with balloons!
                        st.rerun() # Rerun to update the list of known faces in the sidebar

                    else:
                        st.error(f"No face found in the uploaded image for '{new_face_name}'. Please upload an image with a clear face.")
                        
                except Exception as e:
                    st.error(f"Error adding face to Firebase: {e}. "
                                "Check Firebase security rules and network connection.")
        else:
            st.warning("Please provide both a name and upload an image.")

st.markdown("---")
st.markdown("SSO Consultants Face Recognition Tool © 2025 | All Rights Reserved.")
