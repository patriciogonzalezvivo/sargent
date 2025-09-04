#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D   u_image;

uniform vec2    u_resolution;
uniform vec2    u_mouse;
uniform float   u_time;

varying vec4    v_color;
varying vec3    v_normal;
varying vec2    v_texcoord;

void main(void) {
    vec4 color = vec4(vec3(0.0), 1.0);
    vec2 pixel = 1.0/u_resolution.xy;
    vec2 st = gl_FragCoord.xy * pixel;
    vec2 uv = v_texcoord;

#if defined(BACKGROUND)
    color = texture2D(u_image, st);
#else
    color.rgb = v_color.rgb;
#endif

    gl_FragColor = color;
}
