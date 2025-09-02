import argparse
import os 

GLSLVIEWER = "glslViewer"
GLSLVIEWER_ARGS = " --noncurses -l" 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shader"
    )
    parser.add_argument("--scene", type=str, required=True, help="Path to the scene directory")
    parser.add_argument("--shader", type=str, required=True, help="Path to the shader file")
    parser.add_argument("--model", type=str, default="points", help="Type of model to render")
    return parser.parse_args()

def run_shader(scene, shader, model):

    # get image aspect ratio
    img_path = os.path.join(scene, "images")
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        raise FileNotFoundError(f"No image files found in {img_path}")
    
    from PIL import Image
    img = Image.open(os.path.join(img_path, img_files[0]))
    width, height = img.size
    aspect_ratio = width / height
    print(f"Image aspect ratio: {aspect_ratio}")

    # Run glslViewer with the shader and scene data
    width = 512
    height = int(width / aspect_ratio)

    model_path = f"{scene}/sparse/points.ply"
    if model != "points":
        model_path = f"{scene}/{model}.obj"
        model_path += f" {scene}/{model}_texture.png"

    cmd = f"{GLSLVIEWER} shaders/{shader}.vert shaders/{shader}.frag {model_path} {scene}/cameras.csv --u_image {scene}/images -w {width} -h {height} {GLSLVIEWER_ARGS}"
    print(f"Running command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    args = parse_args()
    run_shader(args.scene, args.shader, args.model)
