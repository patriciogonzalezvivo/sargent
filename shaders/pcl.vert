#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex_xyz;        // XYZ
uniform sampler2D u_tex_rgb;        // colors
uniform sampler2D u_tex_normals;    // normals

// uniform vec3    u_scale;
const vec3      u_scale = vec3(0.32324792, 0.69790311, 0.26958888);

uniform mat4    u_modelViewProjectionMatrix;
uniform mat4    u_projectionMatrix;
uniform mat4    u_modelMatrix;
uniform mat4    u_viewMatrix;
uniform mat3    u_normalMatrix;

attribute vec4  a_position;
varying vec4    v_position;


#ifdef MODEL_VERTEX_COLOR
attribute vec4  a_color;
#endif
varying vec4    v_color;

#ifdef MODEL_VERTEX_NORMAL
attribute vec3  a_normal;
#endif
varying vec3    v_normal;

#ifdef MODEL_VERTEX_TEXCOORD
attribute vec2  a_texcoord;
#endif
varying vec2    v_texcoord;

#ifdef LIGHT_SHADOWMAP
uniform mat4    u_lightMatrix;
varying vec4    v_lightCoord;
#endif

void main(void) {
    v_position = u_modelMatrix * a_position;
    
    #ifdef MODEL_VERTEX_TEXCOORD
    v_texcoord = a_texcoord;
    #else
    v_texcoord = a_position.xy * 0.5 + 0.5;
    v_position.xyz = texture2D(u_tex_xyz, v_texcoord).rgb;
    #endif

    #if defined(MODEL_VERTEX_COLOR)
    v_color = a_color;
    #else
    v_color = texture2D(u_tex_rgb, v_texcoord);
    #endif

    #if defined(MODEL_VERTEX_NORMAL)
    v_normal = a_normal;
    #else
    v_normal = (texture2D(u_tex_normals, v_texcoord).rgb * 2.0 - 1.0);
    #endif

    
#ifdef LIGHT_SHADOWMAP
    v_lightCoord = u_lightMatrix * v_position;
#endif
    
    gl_Position = u_projectionMatrix * u_viewMatrix * v_position;
    //gl_Position = u_modelViewProjectionMatrix * v_position;
}
