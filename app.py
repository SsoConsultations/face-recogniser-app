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
        
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}. Please check your .streamlit/secrets.toml and Firebase setup carefully.")
        st.stop()

FIRESTORE_COLLECTION_NAME = st.secrets["firebase"]["firestore_collection"]
STORAGE_KNOWN_FACES_FOLDER = "known_faces_images"

# --- Data Loading Function (Cached for performance) ---
@st.cache_resource(ttl=3600) 
def load_known_faces_from_firebase(_=None): 
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
        
        return known_face_encodings_local, known_face_names_local

    except Exception as e:
        st.error(f"Error loading known faces from Firebase: {e}. "
                 "Ensure your Firestore collection exists and security rules are correct.")
        return [], []

known_face_encodings, known_face_names = load_known_faces_from_firebase()

# --- Face Processing and Drawing Function ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
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
        face_width = right - left
        face_height = bottom - top
        
        base_font_size = 0.002 
        base_thickness = 0.005 
        
        min_font_scale = 0.5
        min_thickness = 1
        
        font_scale = max(min_font_scale, base_font_size * face_width)
        font_thickness = max(min_thickness, int(base_thickness * face_width))
        line_thickness = max(2, int(face_width * 0.01)) 

        box_padding_factor = 0.1 
        box_padding_x = int(face_width * box_padding_factor)
        box_padding_y = int(face_height * box_padding_factor)

        top_ext = max(0, top - box_padding_y)
        right_ext = min(frame_bgr.shape[1], right + box_padding_x)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding_y)
        left_ext = max(0, left - box_padding_x)

        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), line_thickness)

        font = cv2.FONT_HERSHEY_DUPLEX
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        
        label_padding_x = int(text_width * 0.2) 
        label_padding_y = int(text_height * 0.5) 

        label_width = text_width + (label_padding_x * 2)
        label_height = text_height + (label_padding_y * 2) + baseline

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

        text_x = label_left + label_padding_x
        text_y = label_top + label_padding_y + text_height 

        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered", initial_sidebar_state="collapsed")

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'logged_in_as_user' not in st.session_state: 
    st.session_state.logged_in_as_user = False
if 'logged_in_as_admin' not in st.session_state: 
    st.session_state.logged_in_as_admin = False


# --- Home Page ---
if st.session_state.page == 'home':
    # Use a single column to align all content to the left by default
    # Or use st.container() if you want a block without explicit columns
    
    # Removed centering from logo and headings
    try:
        st.image("sso_logo.jpg", width=150) # Logo size from last request
    except FileNotFoundError:
        st.warning("Logo image 'sso_logo.jpg' not found. Please ensure it's in the same directory.")
        st.markdown("## SSO Consultants") 

    # Removed 'text-align: center;' from headings for left alignment
    st.markdown("<h3 style='margin-bottom: 0px;'>SSO Consultants Face Recogniser 🕵️‍♂️</h3>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top: 5px; margin-bottom: 20px; font-size:1.1em;'>Please choose your login type.</p>", unsafe_allow_html=True)

    # Use a single column for buttons to stack them on the left,
    # or use very narrow columns for side-by-side on the left.
    # For a clean left alignment, stacking them might be better than side-by-side with wide gaps.
    # Let's try to stack them.
    # If side-by-side is preferred, you'd use col1, col2 = st.columns([width_for_btn1, width_for_btn2, rest_empty_space])
    
    # Example for stacking buttons left-aligned:
    st.button("Login as User", key="user_login_btn", help="Proceed to face recognition for users")
    st.button("Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities")


# --- User Authentication Page ---
elif st.session_state.page == 'user_auth':
    st.title("User Login 👤")
    st.markdown("Please enter your **username** and **password** to proceed to face recognition.")

    user_username_input = st.text_input("Username:", key="user_username_input")
    user_password_input = st.text_input("Password:", type="password", key="user_password_input")

    if st.button("Login", key="submit_user_login"):
        user_credentials = st.secrets["users"]
        authenticated = False
        for key in user_credentials:
            if key.endswith("_username") and user_credentials[key] == user_username_input:
                password_key = key.replace("_username", "_password")
                if password_key in user_credentials and user_credentials[password_key] == user_password_input:
                    authenticated = True
                    break
        
        if authenticated:
            st.success("User login successful! Redirecting to Face Recognition... 🎉")
            st.session_state.logged_in_as_user = True
            st.session_state.page = 'user_recognition'
            st.rerun()
        else:
            st.error("Invalid username or password for user.")

    if st.button("⬅ Back to Home", key="user_auth_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- User Recognition Page (Accessible only after user login) ---
elif st.session_state.page == 'user_recognition':
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
        else:
            pass

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
        else:
            pass

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
        if admin_username_input == st.secrets["admin"]["username"] and \
           admin_password_input == st.secrets["admin"]["password"]:
            st.success("Admin login successful! Redirecting to Admin Panel... 🎉")
            st.session_state.logged_in_as_admin = True
            st.session_state.page = 'admin_panel'
            st.rerun()
        else:
            st.error("Invalid username or password for admin.")

    if st.button("⬅ Back to Home", key="admin_auth_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Panel (Accessible only after admin login) ---
elif st.session_state.page == 'admin_panel':
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
                    img = Image.open(new_face_image).convert("RGB")
                    img_array = np.array(img)
                    
                    face_locations = face_recognition.face_locations(img_array)
                    face_encodings = face_recognition.face_encodings(img_array, face_locations)

                    if face_encodings:
                        unique_filename = f"{new_face_name.replace(' ', '_').lower()}_{os.urandom(4).hex()}.jpg"
                        storage_path = f"{STORAGE_KNOWN_FACES_FOLDER}/{unique_filename}"
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()

                        blob = st.session_state.bucket.blob(storage_path)
                        blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
                        st.info(f"Image uploaded to Storage: {storage_path}")

                        face_encoding_list = face_encodings[0].tolist() 
                        
                        doc_ref = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).document() 
                        doc_ref.set({
                            "name": new_face_name,
                            "encoding": face_encoding_list,
                            "image_storage_path": storage_path,
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })

                        load_known_faces_from_firebase.clear()
                        known_face_encodings, known_face_names = load_known_faces_from_firebase(_=np.random.rand()) 
                        
                        st.success(f"Successfully added '{new_face_name}' to the known faces database! ✅")
                        st.balloons()
                        st.rerun()

                    else:
                        st.error(f"No face found in the uploaded image for '{new_face_name}'. Please upload an image with a clear face.")
                        
                except Exception as e:
                    st.error(f"Error adding face to Firebase: {e}. "
                             "Check Firebase security rules and network connection.")
        else:
            st.warning("Please provide both a name and upload an image.")

    st.subheader("Current Known Faces 📋")
    if known_face_names:
        unique_names = sorted(list(set(known_face_names)))
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
