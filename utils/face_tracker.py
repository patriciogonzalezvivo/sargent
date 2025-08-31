import numpy as np

import cv2

import mediapipe as mp
from mediapipe import solutions

from .face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis, canonical_uvs, canonical_indices


def generate_preview(landmarks, image: np.ndarray) -> np.ndarray:
    
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    drawing_spec = solutions.drawing_utils.DrawingSpec(
        thickness=1, circle_radius=1, color=(0, 0, 255)
    )

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks,
        connections=solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    return annotated_image


def nodes_faces() -> np.ndarray:
    return np.array(canonical_indices)


def nodes_uvs() -> np.ndarray:
    return np.array(canonical_uvs)


def image_to_nodes(face_img: np.ndarray):
    with solutions.face_mesh.FaceMesh( 
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            # output_face_blendshapes=True,
            # output_facial_transformation_matrixes=True,
        ) as face_mesh:
        results = face_mesh.process(face_img)

        if not results.multi_face_landmarks:
            return None
        
        [face_landmarks] = results.multi_face_landmarks

    # preview: np.ndarray = generate_preview(landmarks=face_landmarks, image=face_img)
    # for landmark in face_landmarks.landmark:
    #     print(landmark)

    nodes = np.stack(
        [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
    )

    # make sure it allways returns 468 landmarks and not 478
    if nodes.shape[0] > 468:
        nodes = nodes[:468, :]

    return nodes


def extract_points(landmarks, indexes, translate=(0,0), scale=(1.0, 1.0)) -> np.ndarray:
    points = np.array([landmarks.landmark[i] for i in indexes])
    points = np.array([[p.x * scale[0] + translate[0], p.y * scale[1] + translate[1]] for p in points])
    return points.astype(np.float32)


def image_to_guidelines(image: np.ndarray, translate: tuple = (0, 0), scale: tuple = (1.0, 1.0)) -> np.ndarray:
    with solutions.face_mesh.FaceMesh( 
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        ) as face_mesh:
        results = face_mesh.process(image)

        if not results.multi_face_landmarks:
            return []
        
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