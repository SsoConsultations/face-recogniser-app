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
    known_face_details_local = [] 
    known_face_docs_local = [] 

    try:
        docs = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).stream()
        for doc in docs:
            face_data = doc.to_dict()
            name = face_data.get("name")
            encoding_list = face_data.get("encoding")
            age = face_data.get("age")
            height = face_data.get("height")
            image_storage_path = face_data.get("image_storage_path")
            
            if name and encoding_list:
                known_face_encodings_local.append(np.array(encoding_list))
                known_face_names_local.append(name)
                known_face_details_local.append({
                    "doc_id": doc.id,
                    "name": name, 
                    "age": age, 
                    "height": height,
                    "image_storage_path": image_storage_path
                })
                known_face_docs_local.append({"id": doc.id, **face_data})
            else:
                st.warning(f"Skipping malformed face data in Firestore document {doc.id}. Missing name or encoding.")
        
        return known_face_encodings_local, known_face_names_local, known_face_details_local, known_face_docs_local

    except Exception as e:
        st.error(f"Error loading known faces from Firebase: {e}. "
                    "Ensure your Firestore collection exists and security rules are correct.")
        return [], [], [], []

known_face_encodings, known_face_names, known_face_details, known_face_docs = load_known_faces_from_firebase()

# --- Face Processing and Drawing Function ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names, known_details):
    frame_rgb_copy = np.copy(frame_rgb)
    
    face_locations = face_recognition.face_locations(frame_rgb_copy)
    face_encodings = face_recognition.face_encodings(frame_rgb_copy, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb_copy, cv2.COLOR_RGB2BGR)

    detected_face_info = [] 

    if not face_locations:
        if 'detected_faces_sidebar_info' in st.session_state:
            del st.session_state['detected_faces_sidebar_info']
        return frame_bgr, []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name_on_image = "Unknown" 
        current_face_details = {"name": "Unknown", "age": "N/A", "height": "N/A"}
        
        if known_encodings: 
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                matched_person_details = known_details[best_match_index]
                name = matched_person_details.get("name", "N/A")
                age = matched_person_details.get("age", "N/A")
                height = matched_person_details.get("height", "N/A")
                
                name_on_image = f"Name: {name}" 
                current_face_details = {"name": name, "age": age, "height": height}
            else:
                if face_distances[best_match_index] < 0.6: 
                    matched_person_details = known_details[best_match_index]
                    name = matched_person_details.get("name", "N/A")
                    age = matched_person_details.get("age", "N/A")
                    height = matched_person_details.get("height", "N/A")
                    
                    name_on_image = f"Possibly {name}"
                    current_face_details = {"name": f"Possibly {name}", "age": age, "height": height}
                else:
                    name_on_image = "Unknown"
                    current_face_details = {"name": "Unknown", "age": "N/A", "height": "N/A"}
        
        detected_face_info.append(current_face_details)

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
        
        (text_width, text_height), baseline = cv2.getTextSize(name_on_image, font, font_scale, font_thickness)
        
        label_padding_x = int(text_width * 0.1) 
        label_padding_y = int(text_height * 0.3) 

        label_width = text_width + (label_padding_x * 2)
        label_height = text_height + (label_padding_y * 2)

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
        text_y = int(label_top + label_padding_y + text_height)
        cv2.putText(frame_bgr, name_on_image, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr, detected_face_info

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered", initial_sidebar_state="expanded") 

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'logged_in_as_user' not in st.session_state: 
    st.session_state.logged_in_as_user = False
if 'logged_in_as_admin' not in st.session_state: 
    st.session_state.logged_in_as_admin = False
if 'detected_faces_sidebar_info' not in st.session_state:
    st.session_state.detected_faces_sidebar_info = []


# --- Home Page ---
if st.session_state.page == 'home':
    
    try:
        st.image("sso_logo.jpg", width=150) 
    except FileNotFoundError:
        st.warning("Logo image 'sso_logo.jpg' not found. Please ensure it's in the same directory.")
        st.markdown("## SSO Consultants") 

    st.markdown("<h3 style='margin-bottom: 0px;'>SSO Consultants Face Recogniser 🕵️‍♂️</h3>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top: 5px; margin-bottom: 20px; font-size:1.1em;'>Please choose your login type.</p>", unsafe_allow_html=True)

    col1_btn, col2_btn = st.columns([0.2, 0.2]) 

    with col1_btn:
        if st.button("Login as User", key="user_login_btn", help="Proceed to face recognition for users"):
            st.session_state.page = 'user_auth' 
            st.rerun()

    with col2_btn:
        if st.button("Login as Admin", key="admin_login_btn", help="Proceed to admin functionalities"):
            st.session_state.page = 'admin_auth' 
            st.rerun()


# --- User Authentication Page ---
elif st.session_state.page == 'user_auth':
    st.title("User Login")
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
            st.success("User login successful! Redirecting to Face Recognition...")
            st.session_state.logged_in_as_user = True
            st.session_state.page = 'user_recognition'
            st.rerun()
        else:
            st.error("Invalid username or password for user.")

    if st.button("Back to Home", key="user_auth_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- User Recognition Page (Accessible only after user login) ---
elif st.session_state.page == 'user_recognition':
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

    with st.sidebar:
        try:
            st.image("sso_logo.jpg", width=150) 
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found in sidebar.")
        st.markdown("---")
        st.header("Choose Input Method")
        option = st.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"), key="user_input_option_sidebar")
        
        st.markdown("---")
        st.header("Recognized Faces Details")
        
        if st.session_state.detected_faces_sidebar_info:
            for i, face_detail in enumerate(st.session_state.detected_faces_sidebar_info):
                st.markdown(f"### Face {i+1}: **{face_detail.get('name', 'Unknown')}**") 
                if face_detail.get('age') != "N/A":
                    st.write(f"**Age:** {face_detail.get('age', 'N/A')}")
                if face_detail.get('height') != "N/A":
                    st.write(f"**Height:** {face_detail.get('height', 'N/A')}")
                st.markdown("---")
        else:
            st.info("No faces detected or recognized yet.")

        st.markdown("---")
        if st.button("Log Out", key="user_logout_sidebar_btn"):
            st.session_state.logged_in_as_user = False
            if 'detected_faces_sidebar_info' in st.session_state:
                del st.session_state['detected_faces_sidebar_info']
            st.session_state.page = 'home'
            st.rerun()

    if option == "Live Webcam Recognition":
        st.subheader("Live Webcam Face Recognition")

        camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="user_camera_input")

        if camera_image is not None:
            with st.spinner("Processing live image..."):
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                processed_img_bgr, detected_faces_info = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names, known_face_details)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                st.session_state['detected_faces_sidebar_info'] = detected_faces_info

            st.image(processed_img_rgb, caption="Processed Live Image", use_container_width=True)
            st.rerun()
        else:
            if 'detected_faces_sidebar_info' in st.session_state:
                del st.session_state['detected_faces_sidebar_info']
            
            if 'prev_camera_image_state' not in st.session_state:
                st.session_state.prev_camera_image_state = None

            if st.session_state.prev_camera_image_state is not None and camera_image is None:
                st.rerun()
            st.session_state.prev_camera_image_state = camera_image


    elif option == "Upload Image for Recognition":
        st.subheader("Upload Image for Face Recognition")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="user_file_uploader")

        if uploaded_file is not None:
            with st.spinner("Loading and processing image..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption="Original Uploaded Image", use_container_width=True)

                processed_img_bgr, detected_faces_info = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names, known_face_details)
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                st.session_state['detected_faces_sidebar_info'] = detected_faces_info

            st.image(processed_img_rgb, caption="Processed Image with Faces", use_container_width=True)
            st.rerun()
        else:
            if 'detected_faces_sidebar_info' in st.session_state:
                del st.session_state['detected_faces_sidebar_info']
            
            if 'prev_uploaded_file_state' not in st.session_state:
                st.session_state.prev_uploaded_file_state = None

            if st.session_state.prev_uploaded_file_state is not None and uploaded_file is None:
                st.rerun()
            st.session_state.prev_uploaded_file_state = uploaded_file

# --- Admin Authentication Page ---
elif st.session_state.page == 'admin_auth':
    st.title("Admin Login")
    st.markdown("Please enter your **admin username** and **password**.")

    admin_username_input = st.text_input("Admin Username:", key="admin_username_input")
    admin_password_input = st.text_input("Password:", type="password", key="admin_pass_input")

    if st.button("Login", key="submit_admin_login"):
        if admin_username_input == st.secrets["admin"]["username"] and \
           admin_password_input == st.secrets["admin"]["password"]:
            st.success("Admin login successful! Redirecting to Admin Panel...")
            st.session_state.logged_in_as_admin = True
            st.session_state.page = 'admin_panel'
            st.rerun()
        else:
            st.error("Invalid username or password for admin.")

    if st.button("Back to Home", key="admin_auth_back_btn"):
        st.session_state.page = 'home'
        st.rerun()

# --- Admin Panel (Accessible only after admin login) ---
elif st.session_state.page == 'admin_panel':
    if not st.session_state.logged_in_as_admin:
        st.warning("Please log in as an admin to access this page.")
        st.session_state.page = 'admin_auth'
        st.rerun()

    with st.sidebar:
        try:
            st.image("sso_logo.jpg", width=150) 
        except FileNotFoundError:
            st.warning("Logo image 'sso_logo.jpg' not found in sidebar.")
        st.markdown("---")
        st.header("Admin Actions")
        admin_option = st.radio("Choose an action:", 
                                ("Add New Face", "View/Update Database"), 
                                key="admin_action_radio")
        st.markdown("---")
        st.header("Faces in Database")
        if known_face_names:
            for name in sorted(list(set(known_face_names))):
                st.write(f"- {name}")
        else:
            st.info("No faces currently registered.")
        st.markdown("---")
        if st.button("Log Out", key="admin_logout_sidebar_btn"):
            st.session_state.logged_in_as_admin = False
            st.session_state.page = 'home'
            st.rerun()

    st.title("Admin Panel")
    st.markdown("This section is for **administrators** only.")

    if admin_option == "Add New Face":
        st.subheader("Add New Face to Database")
        st.markdown("Upload an image of a person and provide a name and details for recognition.")

        new_face_name = st.text_input("Enter Name/Description for the Face:", key="new_face_name_input")
        new_face_age = st.number_input("Enter Age (optional):", min_value=0, max_value=150, value=None, format="%d", key="new_face_age_input")
        new_face_height = st.text_input("Enter Height (e.g., 5'10\" or 178cm) (optional):", key="new_face_height_input")
        
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
                            doc_data = {
                                "name": new_face_name,
                                "encoding": face_encoding_list,
                                "image_storage_path": storage_path,
                                "timestamp": firestore.SERVER_TIMESTAMP
                            }
                            if new_face_age is not None:
                                doc_data["age"] = new_face_age
                            if new_face_height:
                                doc_data["height"] = new_face_height

                            doc_ref.set(doc_data)

                            load_known_faces_from_firebase.clear()
                            known_face_encodings, known_face_names, known_face_details, known_face_docs = load_known_faces_from_firebase(_=np.random.rand()) 
                            
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

    # --- View/Update Database Section ---
    elif admin_option == "View/Update Database":
        st.subheader("View and Update Face Database")
        st.markdown("Browse registered faces. You can update their details or re-upload their image.")

        if not known_face_docs:
            st.info("No faces are currently registered in the database.")
        else:
            import pandas as pd
            display_data = []
            for doc in known_face_docs:
                display_data.append({
                    "Name": doc.get("name", "N/A"),
                    "Age": doc.get("age", "N/A"),
                    "Height": doc.get("height", "N/A"),
                })
            df = pd.DataFrame(display_data)

            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader("Update Existing Face Details")
            st.markdown("Select a face from the dropdown to update its information.")

            name_to_id_map = {f"{doc.get('name', 'Unnamed')} (ID: {doc['id'][:6]}...)": doc["id"] for doc in known_face_docs}
            
            selected_face_label = st.selectbox(
                "Select a person to update:",
                options=list(name_to_id_map.keys()),
                key="select_face_to_update"
            )

            if selected_face_label:
                selected_doc_id = name_to_id_map[selected_face_label]
                selected_doc = next(doc for doc in known_face_docs if doc["id"] == selected_doc_id)

                st.write(f"**Currently updating:** {selected_doc.get('name', 'Unnamed')}")

                if selected_doc.get("image_storage_path"):
                    try:
                        blob = st.session_state.bucket.blob(selected_doc["image_storage_path"])
                        image_bytes = blob.download_as_bytes()
                        st.image(image_bytes, caption=f"Current image for {selected_doc.get('name', 'Unnamed')}", width=200)
                    except Exception as e:
                        st.warning(f"Could not load image from storage: {e}")

                updated_name = st.text_input("New Name:", value=selected_doc.get("name", ""), key=f"update_name_{selected_doc_id}")
                updated_age = st.number_input("New Age (optional):", min_value=0, max_value=150, value=selected_doc.get("age"), format="%d", key=f"update_age_{selected_doc_id}")
                updated_height = st.text_input("New Height (optional):", value=selected_doc.get("height", ""), key=f"update_height_{selected_doc_id}")
                re_upload_image = st.file_uploader("Re-upload Image (optional, will replace current image):", 
                                                    type=["jpg", "jpeg", "png"], 
                                                    key=f"re_upload_image_{selected_doc_id}")

                col_update, col_delete = st.columns(2)

                with col_update:
                    if st.button(f"Update {updated_name}", key=f"update_btn_{selected_doc_id}"):
                        with st.spinner("Updating face details..."):
                            try:
                                update_data = {
                                    "name": updated_name,
                                    "age": updated_age,
                                    "height": updated_height
                                }

                                new_encoding = None
                                new_storage_path = selected_doc.get("image_storage_path")

                                if re_upload_image is not None:
                                    img = Image.open(re_upload_image).convert("RGB")
                                    img_array = np.array(img)
                                    face_locations = face_recognition.face_locations(img_array)
                                    face_encodings = face_recognition.face_encodings(img_array, face_locations)

                                    if face_encodings:
                                        if selected_doc.get("image_storage_path"):
                                            try:
                                                old_blob = st.session_state.bucket.blob(selected_doc["image_storage_path"])
                                                if old_blob.exists():
                                                    old_blob.delete()
                                                    st.info(f"Deleted old image: {selected_doc['image_storage_path']}")
                                            except Exception as e:
                                                st.warning(f"Could not delete old image from storage: {e}")

                                        unique_filename = f"{updated_name.replace(' ', '_').lower()}_{os.urandom(4).hex()}.jpg"
                                        new_storage_path = f"{STORAGE_KNOWN_FACES_FOLDER}/{unique_filename}"
                                        img_byte_arr = io.BytesIO()
                                        img.save(img_byte_arr, format='JPEG')
                                        img_byte_arr = img_byte_arr.getvalue()

                                        new_blob = st.session_state.bucket.blob(new_storage_path)
                                        new_blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
                                        st.info(f"New image uploaded to Storage: {new_storage_path}")
                                        
                                        new_encoding = face_encodings[0].tolist()
                                        update_data["encoding"] = new_encoding
                                        update_data["image_storage_path"] = new_storage_path
                                    else:
                                        st.warning("No face found in the re-uploaded image. Face encoding and image path will not be updated.")
                                        
                                doc_ref = st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).document(selected_doc_id)
                                doc_ref.update(update_data)

                                load_known_faces_from_firebase.clear()
                                known_face_encodings, known_face_names, known_face_details, known_face_docs = load_known_faces_from_firebase(_=np.random.rand())
                                
                                st.success(f"Successfully updated '{updated_name}'! ✅")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error updating face: {e}")
                                
                with col_delete:
                    # Added a "Confirm Delete" step for safety
                    if st.button(f"Delete {selected_doc.get('name', 'Unnamed')}", key=f"delete_btn_{selected_doc_id}"):
                        st.warning("Are you sure you want to delete this entry? This action is irreversible.")
                        if st.button("Confirm Delete", key=f"confirm_delete_{selected_doc_id}"):
                            with st.spinner(f"Deleting {selected_doc.get('name', 'Unnamed')}..."):
                                try:
                                    if selected_doc.get("image_storage_path"):
                                        try:
                                            blob = st.session_state.bucket.blob(selected_doc["image_storage_path"])
                                            if blob.exists():
                                                blob.delete()
                                                st.info(f"Deleted image from Storage: {selected_doc['image_storage_path']}")
                                        except Exception as e:
                                            st.warning(f"Could not delete image from storage: {e}")

                                    st.session_state.db.collection(FIRESTORE_COLLECTION_NAME).document(selected_doc_id).delete()

                                    load_known_faces_from_firebase.clear()
                                    known_face_encodings, known_face_names, known_face_details, known_face_docs = load_known_faces_from_firebase(_=np.random.rand())

                                    st.success(f"Successfully deleted '{selected_doc.get('name', 'Unnamed')}'! 🗑️")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error deleting face: {e}")

st.markdown("---")
st.markdown("SSO Consultants Face Recognition Tool © 2025 | All Rights Reserved.")
