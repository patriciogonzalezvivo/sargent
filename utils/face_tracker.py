import numpy as np

import cv2

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis

# import pyvista as pv

# # convert landmarks to image
# def landmarks2image(rgb_image, detection_result):
#   face_landmarks_list = detection_result.face_landmarks
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected faces to visualize.
#   for idx in range(len(face_landmarks_list)):
#     face_landmarks = face_landmarks_list[idx]

#     # Draw the face landmarks.
#     face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     face_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#     ])

#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_tesselation_style())
    
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_contours_style())
    
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp.solutions.drawing_styles
#           .get_default_face_mesh_iris_connections_style())

#   return annotated_image


# def generate_preview(landmarks, image: np.ndarray) -> np.ndarray:
    
#     annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
#         thickness=1, circle_radius=1, color=(0, 0, 255)
#     )

#     mp.solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=landmarks,
#         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec=drawing_spec,
#         connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
#     )

#     return annotated_image


class NoFacesFoundException(Exception):
    pass


# def face2mesh(face_img: np.ndarray) -> Tuple[pv.PolyData, np.ndarray]:


#     points_idx = [33,263,61,291,199]
#     points_idx = points_idx + [key for (key,val) in procrustes_landmark_basis]
#     points_idx = list(set(points_idx))
#     points_idx.sort()

#     frame_height, frame_width, _ = face_img.shape
#     focal_length = frame_width
#     center = (frame_width/2, frame_height/2)
#     camera_matrix = np.array(
#                             [[focal_length, 0, center[0]],
#                             [0, focal_length, center[1]],
#                             [0, 0, 1]], dtype = "double"
#                             )

#     pcf = PCF(near=1,far=10000,frame_height=frame_height,frame_width=frame_width,fy=camera_matrix[1,1])
#     dist_coeff = np.zeros((4, 1))

#     with mp.solutions.face_mesh.FaceMesh( 
#         static_image_mode=True, 
#         max_num_faces=1, 
#         refine_landmarks=True, 
#         min_detection_confidence=0.5, 
#         # output_facial_transformation_matrixes=True,
#         ) as face_mesh:
#         results = face_mesh.process(face_img)

#         if not results.multi_face_landmarks:
#             raise NoFacesFoundException()
        
#         [face_landmarks] = results.multi_face_landmarks

#     preview: np.ndarray = generate_preview(landmarks=face_landmarks, face_img=face_img)
#     nodes = np.stack(
#         [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
#     )

#     landmarks = np.array([(lm.x,lm.y,lm.z) for lm in face_landmarks.landmark])
#     landmarks = landmarks.T
#     metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
#     model_points = metric_landmarks[0:3, points_idx].T
#     image_points = landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None,:]

#     success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeff, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
#     # _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeff)
#     (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeff)

#     print("Rotation Vector:\n {0}".format(rotation_vector))
#     print("Translation Vector:\n {0}".format(translation_vector))
#     print("Nose End Point:\n {0}".format(nose_end_point2D))
#     print("Pose Transform Matrix:\n {0}".format(pose_transform_mat))
#     print("Camera Matrix:\n {0}".format(camera_matrix))
#     print("Distortion Coefficients:\n {0}".format(dist_coeff))

#     def edges2faces(edges: np.array):
#         """Construct properly oriented faces from the list of edges"""
#         faces = []
#         cycles = []

#         for i, (u1, v1) in enumerate(edges):
#             mask = edges[:, 0] == v1

#             next_idx = np.argwhere(mask).flatten()
#             next_edges = edges[mask]

#             for j, (u2, v2) in zip(next_idx, next_edges):
#                 mask = (edges[:, 0] == v2) & (edges[:, 1] == u1)

#                 if mask.sum() == 1:
#                     last_idx = np.argwhere(mask).item()

#                     if {i, j, last_idx} not in cycles:
#                         faces.append((u1, v1, v2))
#                         cycles.append({i, j, last_idx})

#         return np.array(faces)

#     edges = np.array(list(mp.solutions.face_mesh.FACEMESH_TESSELATION))
#     faces = edges2faces(edges)

#     edges = np.c_[2 * np.ones(len(edges))[:, None], edges].flatten().astype(int)
#     faces = np.c_[3 * np.ones(len(faces))[:, None], faces].flatten().astype(int)

#     poly = pv.PolyData(nodes, faces=faces, lines=edges)
#     return poly, preview

def extract_points(landmarks, indexes, translate=(0,0), scale=(1.0, 1.0)) -> np.ndarray:
    points = np.array([landmarks.landmark[i] for i in indexes])
    points = np.array([[p.x * scale[0] + translate[0], p.y * scale[1] + translate[1]] for p in points])
    return points.astype(np.float32)

def image_to_guidelines(image: np.ndarray, translate: tuple = (0, 0), scale: tuple = (1.0, 1.0)) -> np.ndarray:
    with mp.solutions.face_mesh.FaceMesh( 
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        # output_facial_transformation_matrixes=True,
        ) as face_mesh:
        results = face_mesh.process(image)

        if not results.multi_face_landmarks:
            raise NoFacesFoundException()
        
        [face_landmarks] = results.multi_face_landmarks

    guidelines = []

    # # jaw_line
    # jaw_line_indeces = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356]
    # guidelines.append(extract_points(face_landmarks, jaw_line_indeces, translate, scale))

    # middle line
    # middle_line_indeces = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
    middle_line_indeces = [10, 9, 168, 6, 4, 1, 13, 17, 152]
    guidelines.append(extract_points(face_landmarks, middle_line_indeces, translate, scale))

    # eyebrow line
    eyebrow_indeces = [70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300]
    # eyebrow_indeces = [70, 63, 66, 9, 296, 293, 300] 
    guidelines.append(extract_points(face_landmarks, eyebrow_indeces, translate, scale))

    # bottom of the nose
    nose_indeces = [97, 2, 326]
    guidelines.append(extract_points(face_landmarks, nose_indeces, translate, scale))

    # top lip line
    top_lip_indeces = [78, 13, 308]
    guidelines.append(extract_points(face_landmarks, top_lip_indeces, translate, scale))

    # bottom lip line
    bottom_lip_indeces = [84, 17, 314]
    guidelines.append(extract_points(face_landmarks, bottom_lip_indeces, translate, scale))


    return guidelines