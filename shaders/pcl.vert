#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0; // XYZ
uniform sampler2D u_tex1; // colors
uniform sampler2D u_tex2; // normals

// uniform vec3    u_scale;
const vec3      u_scale = vec3(0.32324792, 0.69790311, 0.26958888);

uniform mat4    u_modelViewProjectionMatrix;
uniform mat4    u_projectionMatrix;
uniform mat4    u_modelMatrix;
uniform mat4    u_viewMatrix;
uniform mat3    u_normalMatrix;

attribute vec4  a_position;
varying vec4    v_position;

varying vec4    v_color;
varying vec3    v_normal;
varying vec2    v_texcoord;

#ifdef LIGHT_SHADOWMAP
uniform mat4    u_lightMatrix;
varying vec4    v_lightCoord;
#endif

void main(void) {
    v_position = u_modelMatrix * a_position;
    v_texcoord = a_position.xy * 0.5 + 0.5;

    vec3 xyz = texture2D(u_tex0, v_texcoord).rgb;// * u_scale;
    vec3 rgb = texture2D(u_tex1, v_texcoord).rgb;
    v_position.xyz = xyz * 2.0 - 1.0;
    v_normal = (texture2D(u_tex2, v_texcoord).rgb * 2.0 - 1.0);
    v_color = vec4(rgb, 1.0);
    
#ifdef LIGHT_SHADOWMAP
    v_lightCoord = u_lightMatrix * v_position;
#endif
    
    gl_Position = u_projectionMatrix * u_viewMatrix * v_position;
    //gl_Position = u_modelViewProjectionMatrix * v_position;
}
