
#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D   u_image;
uniform sampler2D   u_tex0;

uniform sampler2D   u_scene;
uniform sampler2D   u_sceneDepth;

uniform mat4        u_projectionMatrix;

uniform vec3        u_camera;
uniform float       u_cameraNearClip;
uniform float       u_cameraFarClip;

uniform vec3        u_light;
uniform vec3        u_lightColor;
uniform float       u_lightFalloff;
uniform float       u_lightIntensity;

uniform float       u_iblLuminance;

uniform samplerCube u_cubeMap;
uniform vec3        u_SH[9];

#ifdef LIGHT_SHADOWMAP
uniform sampler2D   u_lightShadowMap;
uniform mat4        u_lightMatrix;
varying vec4        v_lightCoord;
#endif

uniform vec2        u_resolution;
uniform float       u_time;

varying vec4        v_position;
varying vec4        v_color;
varying vec3        v_normal;

#ifdef MODEL_VERTEX_TEXCOORD
varying vec2        v_texcoord;
#endif

#ifdef MODEL_VERTEX_TANGENT
varying vec4        v_tangent;
varying mat3        v_tangentToWorld;
#endif

void main(void) {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    vec2 pixel = 1.0/u_resolution;
    vec2 st = gl_FragCoord.xy * pixel;
    

#if defined(BACKGROUND)
    color = texture2D(u_image, st);

#else 

    vec2 uv = v_texcoord;
    uv.y = 1.0 - uv.y;
    color = texture2D(u_tex0, uv);

// #ifdef MODEL_VERTEX_TEXCOORD
//     // color.rg = v_texcoord;
//     vec2 uv = v_texcoord;
//     uv.y = 1.0 - uv.y;
//     color = texture2D(u_tex0, uv);
// #endif

#endif

    gl_FragColor = color;
}
